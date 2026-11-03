for k in range(2):
    for i in range(2):
        W[(k + 2) * n + i, col:col+7] = W2[k * 2 + i, :]
