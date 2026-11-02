import numpy as np
import itertools

DIMENSIONS = [4, 5, 6]

def construct():
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Initialize lists for U, V, W
    # U shape: (n*m, rank)
    # V shape: (m*p, rank)
    # W shape: (p*n, rank)
    
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    # The trivial decomposition uses the standard basis tensors.
    # Each term corresponds to a unique (i, j, k) triplet.
    # We map the triplet (i, j, k) to a column index col.
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # U[i*m + j, col] = 1
                U[i * m + j][col] = 1.0
                # V[j*p + k, col] = 1
                V[j * p + k][col] = 1.0
                # W[k*n + i, col] = 1
                W[k * n + i][col] = 1.0
                col += 1
                
    return {"U": U, "V": V, "W": W}
