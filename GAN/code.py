import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import sys
import PIL
from IPython.display import Image
import tensorflow as tf
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import random
from sklearn.model_selection import train_test_split

def resize_images():
    for file_name in os.listdir("./CAT - Dataset/CAT_00/"):
        if not file_name.endswith(".cat"):
            img = cv2.imread("./CAT - Dataset/CAT_00/" + file_name, cv2.IMREAD_COLOR)
            try:
                img = cv2.resize(img, (128, 128), interpolation = cv2.INTER_AREA)
                cv2.imwrite("./filtered/" + file_name, img)
            except:
                print("Arquivo " + file_name + " falhou!")

def generate_data(amount_samples = 20):
    amount_samples = 20
    X = []
    y = []

    index = 0
    for file_name in os.listdir("./filtered/"):
        if index == amount_samples - 1:
            break
        
        img = tf.keras.utils.load_img("./filtered/" + file_name)
        img = np.asarray(img)
        X.append(img)
        y.append(1)
        index += 1

    # Testing random noise
    for _ in range(amount_samples):
        noise = np.random.uniform(low=0, high=1, size=(128, 128, 3))
        noise = np.asarray(noise)
        X.append(noise)
        y.append(0)

    return (X, y)

NUM_COLORS = 3
IMAGE_SIZE = 128

# Define model
# class Generator(nn.Module):
#     def __init__(self):
#         super(Generator, self).__init__()
#         self.main = nn.Sequential(
#             nn.Conv2d(NUM_COLORS, IMAGE_SIZE, 4, 2, 1, bias=False),
#             nn.Sigmoid()
#         )

#     def forward(self, input):
#         return self.main(input)

discriminator = models.Sequential()
discriminator.add(layers.Conv2D(32, (32, 32), activation='relu', input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3)))
discriminator.add(layers.MaxPooling2D((2, 2)))
discriminator.add(layers.Conv2D(64, (16, 16), activation='relu'))
discriminator.add(layers.MaxPooling2D((2, 2)))
discriminator.add(layers.Flatten())
discriminator.add(layers.Dense(128, activation='relu'))
discriminator.add(layers.Dense(1, activation='sigmoid'))
discriminator.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

#for i in range(20):
#     # zero the gradients on each iteration
#     generator_optimizer.zero_grad()

#     # Create noisy input for generator
#     # Need float type instead of int
#     noise = torch.randint(0, 2, size=(3, 128, 128)).float()
#     noise = noise.to(device)
#     #noise = torch.randint(0, 2, size=(batch_size, input_length)).float()
#     generated_data = generator(noise)

    # Generate examples of even real data
    #true_labels, true_data = generate_even_data(max_int, batch_size=batch_size)
    #true_labels = torch.tensor(true_labels).float()
    #true_data = torch.tensor(true_data).float()

    # Train the generator
    # We invert the labels here and don't train the discriminator because we want the generator
    # to make things the discriminator classifies as true.
    #generator_discriminator_out = discriminator(generated_data)
    
#img = cv2.imread("./filtered/00000001_000.jpg", cv2.IMREAD_COLOR)

if __name__ == '__main__':
    X, y = generate_data()
    datagen = ImageDataGenerator(rescale=1.0/255.0)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_test = np.array(X_test)
    y_test = np.array(y_test)

    train_iterator = datagen.flow(X_train, y_train, batch_size=5)
    test_iterator = datagen.flow(X_test, y_test, batch_size=5)

    #discriminator.fit(X_train, y_train, epochs=20, batch_size=1, verbose=1) 

    discriminator.fit_generator(train_iterator, steps_per_epoch=len(train_iterator), epochs=20)
    _, acc = discriminator.evaluate_generator(test_iterator, steps=len(test_iterator), verbose=0)
    print('Test Accuracy: %.3f' % (acc * 100))