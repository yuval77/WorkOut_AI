from exercise_data import *
from DL3 import *
import numpy as np
import matplotlib.pyplot as plt
import random

crunches1, crunches2, legs90, plank, pull_ups1, pull_ups2, push_ups1, push_ups2, resting, squat1, squat2, wall_sit = exercise_data()
x_data = crunches1 + crunches2 + legs90 + plank + pull_ups1 + pull_ups2 + push_ups1 + push_ups2 + resting + squat1 + squat2 + wall_sit
y_data = []
for i in range(len(crunches1)):
    y_data.append(0)
for i in range(len(crunches2)):
    y_data.append(1)
for i in range(len(legs90)):
    y_data.append(2)
for i in range(len(plank)):
    y_data.append(3)
for i in range(len(pull_ups1)):
    y_data.append(4)
for i in range(len(pull_ups2)):
    y_data.append(5)
for i in range(len(push_ups1)):
    y_data.append(6)
for i in range(len(push_ups2)):
    y_data.append(7)
for i in range(len(resting)):
    y_data.append(8)
for i in range(len(squat1)):
    y_data.append(9)
for i in range(len(squat2)):
    y_data.append(10)
for i in range(len(wall_sit)):
    y_data.append(11)

x_train = []
x_test = []
y_train = []
y_test = []

counter_20 = 0
for i in range(12592):
    counter_20 += 1
    rnd = random.randint(0, len(y_data) - 1)
    print(str(rnd) + " out of " + str(len(x_train)))

    if counter_20 % 20 != 0:
        x_train.append(x_data[rnd])
        y_train.append(y_data[rnd])
    else:
        x_test.append(x_data[rnd])
        y_test.append(y_data[rnd])

    x_data.remove(x_data[rnd])
    y_data.remove(y_data[rnd])

# adjusting the data:
np_x_train = np.array(x_train)
x_train = np_x_train.transpose() / 64 - 2
np_y_train = np.array(y_train)
y_train = np.zeros((1, 11962))
for i in range(11963):
    y_train[0][i] = np_y_train[0][i]

np_x_test = np.array(x_test)
x_test = np_x_test.transpose() / 64 - 2
np_y_test = np.array(y_test)
y_test = np.zeros((1, 629))
for i in range(630):
    y_train[0][i] = np_y_train[0][i]

softmax_layer = DLLayer("Softmax 3", 12, (24,), "softmax")
model = DLModel()
model.add(softmax_layer)
model.compile("categorical_cross_entropy")

X = x_train
Y = y_train

costs = model.train(X, Y, 1000)
plt.plot(costs)
plt.show()
predictions = model.predict(X)
print("right", np.sum(Y.argmax(axis=0) == predictions.argmax(axis=0)))
print("wrong", np.sum(Y.argmax(axis=0) != predictions.argmax(axis=0)))

