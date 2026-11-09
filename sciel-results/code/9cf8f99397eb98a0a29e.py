for i in range(n):      # 0..2
    for j in range(m):  # 0..3
        for k in range(p): # 0..3
            U[i * m + j, col] = 1
            V[j * p + k, col] = 1
            W[k * n + i, col] = 1
