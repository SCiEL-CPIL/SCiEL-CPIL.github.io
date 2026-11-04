def construct():
    n, m, p = 3, 3, 5
    rank = m * p  # = 15, which is the minimal rank
    # V is the identity matrix of size (m*p) x rank
    V = [[0] * rank for _ in range(m * p)]
    for j in range(m):
        for k in range(p):
            V[j * p + k][j * p + k] = 1
    # U[i*m+j, c] = 1 if c in [j*p, (j+1)*p)
    U = [[0] * rank for _ in range(n * m)]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][j * p + k] = 1
    # W[k*n+i, c] = 1 if c % p == k
    W = [[0] * rank for _ in range(p * n)]
    for k in range(p):
        for i in range(n):
            for j in range(m):
                W[k * n + i][j * p + k] = 1
    return {"U": U, "V": V, "W": W}
