import numpy as np

DIMENSIONS = [3, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    # Rank n*m*p decomposition: trivial exact decomposition.
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # U picks A[i, j]
                U[i * m + j][col] = 1
                # V picks B[j, k]
                V[j * p + k][col] = 1
                # W picks C[i, k] (index k*n + i)
                W[k * n + i][col] = 1
                col += 1
                
    return {"U": U, "V": V, "W": W}
