import numpy as np
import time

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Returns a valid tensor decomposition for 5x5 matrix multiplication.
    Uses schoolbook decomposition with rank 125, which is guaranteed valid.
    
    The decomposition satisfies:
    sum_r U[a, r] * V[b, r] * W[c, r] == T[a, b, c]
    
    Indexing convention (as specified):
    U index: i * m + j  (for A[i,j])
    V index: j * p + k  (for B[j,k])  
    W index: k * n + i  (for C[i,k], column-major flat index)
    """
    n, m, p = DIMENSIONS
    rank = n * m * p
    
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
