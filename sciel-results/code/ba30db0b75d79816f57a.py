"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 9]

import time
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    """
    Return the exact tensor decomposition for 2x3 * 3x9 matrix multiplication.
    
    The matrix multiplication tensor T[i,j,k] = A[i,j] * B[j,k] for i in {0,1},
    j in {0,1,2}, k in {0,...,8} has rank 54 (schoolbook).
    
    U has shape (n*m, rank) = (6, 54): U[i*m + j, c]
    V has shape (m*p, rank) = (27, 54): V[j*p + k, c]
    W has shape (p*n, rank) = (18, 54): W[k*n + i, c]
    
    For each column c corresponding to (i,j,k):
      U[i*m+j, c] = 1
      V[j*p+k, c] = 1
      W[k*n+i, c] = 1
    """
    n, m, p = DIMENSIONS
    rank = n * m * p  # 54
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
