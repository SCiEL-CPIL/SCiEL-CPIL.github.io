"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 6]

import time
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    """
    Returns a tensor decomposition for multiplying a 3x3 matrix by a 3x6 matrix.
    Uses the schoolbook algorithm with rank n*m*p = 54.
    U: (9, R), V: (18, R), W: (18, R)
    W index convention: W[k*n + i, r]
    """
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Initialize matrices as lists of lists
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # U[i*m + j, column] = 1
                U[i * m + j][column] = 1
                # V[j*p + k, column] = 1
                V[j * p + k][column] = 1
                # W[k*n + i, column] = 1 (Official trace convention)
                W[k * n + i][column] = 1
                column += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
