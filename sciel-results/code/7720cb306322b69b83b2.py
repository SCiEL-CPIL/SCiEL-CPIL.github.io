for i in range(n):
    for k in range(p):
        r = i * p + k
        for j in range(m):
            U[i * m + j, r] = 1.0
        for jj in range(m):
            V[jj * p + k, r] = 1.0
        W[k * n + i, r] = 1.0
