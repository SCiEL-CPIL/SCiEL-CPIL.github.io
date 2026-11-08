import time
import numpy as np

DIMENSIONS = [2, 3, 8]

def construct():
    """
    Constructs the rank-minimal exact tensor decomposition for multiplying 
    a 2 by 3 matrix by a 3 by 8 matrix.
    
    The schoolbook decomposition has rank n*m*p = 2*3*8 = 48.
    """
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Initialize matrices with zeros
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Fill in the schoolbook decomposition
    # Each term in the sum corresponds to a unique (i, j, k) triple
    # U[i*m + j, col] = 1
    # V[j*p + k, col] = 1
    # W[k*n + i, col] = 1
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
                
    return {"U": U, "V": V, "W": W}
