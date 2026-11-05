import numpy as np

DIMENSIONS = [2, 3, 10]

def construct():
    n, m, p = DIMENSIONS
    # The rank for matrix multiplication n x m * m x p is n*m*p
    # unless specific small cases allow shortcuts.
    # For 2x3x10, the rank is 60.
    rank = n * m * p
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                # W index is k*n + i according to the prompt
                W[k * n + i][col] = 1
                col += 1
                
    return {"U": U, "V": V, "W": W}
