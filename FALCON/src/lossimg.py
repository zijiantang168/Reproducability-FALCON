import matplotlib.pyplot as plt
import numpy as np

fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12,6))

X_acc = list()
X_loss = list()
y = list()

f1 = '../VGG19 -conv FALCON -init -data cifar10 -train.txt'
f1_title = 'VGG19'
f2 = '../ResNet -conv FALCON -init -data cifar10 -train.txt'
f2_title = 'ResNet34'

f3 = '../ResNet -conv StandardConv -init -data cifar10 -train.txt'
f3_title = 'ResNet StandardConv'

f4 = '../VGG19 -conv StandardConv -init -data cifar10 -train.txt'
f4_title='VGG19 StandardConv'

# with open('../VGG19 -conv FALCON -init -data cifar10 -train.txt', 'r') as f:
with open(f4, 'r') as f:
    for line in f:
        array = line.split(' ')
        X_acc.append(float(array[0]))
        X_loss.append(float(array[1]))
        y.append(float(array[2]))

y = np.arange(0, 350, 350 / len(y))

ax[0].plot(y, X_acc)
ax[1].plot(y, X_loss)
ax[0].set_title(f4_title)
ax[1].set_title(f4_title)

ax[0].set_xlabel('epochs')
ax[1].set_xlabel('epochs')
ax[0].set_ylabel('accuracy')
ax[1].set_ylabel('loss')
for i in range(2):
    # ax[i].legend()
    ax[i].grid(True)


plt.show()