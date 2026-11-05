import numpy as np

DIMENSIONS = [3, 3, 7]

def construct():
    n, m, p = DIMENSIONS
    rank = n
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for r in range(rank):
        # U[i*m + j, r] = 1 for all i, j
        for i in range(n):
            for j in range(m):
                U[i * m + j][r] = 1
                
        # V[j*p + k, r] = 1 for all j, k
        for j in range(m):
            for k in range(p):
                V[j * p + k][r] = 1
                
        # W[k*n + i, r] = 1 if k == r and i == r, else 0
        for k in range(p):
            for i in range(n):
                if k == r and i == r:
                    W[k * n + i][r] = 1
                else:
                    W[k * n + i][r] = 0
                    
    return {"U": U, "V": V, "W": W}
