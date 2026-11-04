for j in range(m):
    for k in range(p):
        U[0 * m + j][col] = 1
        V[j * p + k][col] = 1
        W[k * n + 0][col] = 1
        col += 1
