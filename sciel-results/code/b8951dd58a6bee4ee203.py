import time

DIMENSIONS = [2, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    rank = m  # The minimum rank is m
    
    # U: (n*m, rank) = (8, 4)
    # U[i*m + j, t] = 1 if j == t, else 0
    U = [[0] * rank for _ in range(n * m)]
    for i in range(n):
        for j in range(m):
            U[i * m + j][j] = 1
    
    # V: (m*p, rank) = (28, 4)
    # V[j*p + k, t] = 1 if j == t, else 0
    V = [[0] * rank for _ in range(m * p)]
    for j in range(m):
        for k in range(p):
            V[j * p + k][j] = 1
    
    # W: (p*n, rank) = (14, 4)
    # W[k*n + i, t] = 1 for all i, k, t
    W = [[1] * rank for _ in range(p * n)]
    
    return {"U": U, "V": V, "W": W}
