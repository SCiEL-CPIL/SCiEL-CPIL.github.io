for i in range(n):
    for j in range(m):
        col = i * m + j
        U[i * m + j, col] = 1.0
        for k in range(p):
            V[j * p + k, col] = 1.0
        for k in range(p):
            W[k * n + i, col] = 1.0 / m
