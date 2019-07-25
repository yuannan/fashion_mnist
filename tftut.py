from __future__ import absolute_import, division, print_function, unicode_literals

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt
import os

model_path = "model.h5"

# importing mnist data set
fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# normanlizing the data to
train_images = train_images / 255.0
test_images = test_images / 255.0

def create_model():
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(28, 28)),
        keras.layers.Dense(128, activation=tf.nn.relu),
        keras.layers.Dense(10, activation=tf.nn.softmax)
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def train():
    global model
    # building model

    model = create_model()

    ep = int(input("epochs: "))
    # start training
    model.fit(train_images, train_labels,  epochs = ep)

    # evaluate accuracy
    test_loss, test_acc = model.evaluate(test_images, test_labels)
    print('Test accuracy:', test_acc, test_loss)

def plot_image(i, predictions_array, true_label, img):
    predictions_array, true_label, img = predictions_array[i], true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img, cmap=plt.cm.binary)

    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        color = 'blue'
    else:
        color = 'red'

    plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                100*np.max(predictions_array),
                                class_names[true_label]),
                                color=color)

def plot_value_array(i, predictions_array, true_label):
    predictions_array, true_label = predictions_array[i], true_label[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    thisplot = plt.bar(range(10), predictions_array, color="#777777")
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)

    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')

def show_results():
    # testing results
    predictions = model.predict(test_images)

    num_rows = 5
    num_cols = 3
    num_images = num_rows*num_cols
    plt.figure(figsize=(2*2*num_cols, 2*num_rows))
    for i in range(num_images):
      plt.subplot(num_rows, 2*num_cols, 2*i+1)
      plot_image(i, predictions, test_labels, test_images)
      plt.subplot(num_rows, 2*num_cols, 2*i+2)
      plot_value_array(i, predictions, test_labels)
    plt.show()

def show_test_img(i):
    predictions = model.predict(test_images)
    plt.figure(figsize=(6,3))
    plt.subplot(1,2,1)
    plot_image(i, predictions, test_labels, test_images)
    plt.subplot(1,2,2)
    plot_value_array(i, predictions,  test_labels)
    plt.show()

def show_result():
    img_i = input("Image index: ")
    show_test_img(int(img_i))

def import_model():
    print("import weights")
    global model

    model = create_model()
    loss, acc = model.evaluate(test_images, test_labels)
    print("Untrained model, accuracy: {:5.2f}%".format(100*acc))

    model = keras.models.load_model(model_path)
    loss,acc = model.evaluate(test_images, test_labels)
    print("Restored model, accuracy: {:5.2f}%".format(100*acc))

def export_model():
    print("exporting to " + model_path)
    model.save(model_path)

def menu():
    options = { 1: train,
                2: import_model,
                3: export_model,
                4: show_results,
                5: show_result}

    inmenu = True
    while(inmenu):
        print("1) Train Model\n2) Import\n3) Export\n4) Show results\n5) Show result\nOther to exit")

        menuinput = input()
        menuint = int(menuinput)
        if(menuint in range(1,6)):
            options[menuint]()
        else:
            print("not in range, exiting")
            inmenu = False

if __name__ == "__main__":
    menu()
