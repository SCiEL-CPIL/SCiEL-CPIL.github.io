import time
import numpy as np
import sys

DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Returns an exact tensor decomposition for multiplying a 3x3 matrix by a 3x5 matrix.
    Uses schoolbook multiplication with rank 45, which is the standard baseline.
    """
    n, m, p = DIMENSIONS
    rank = n * m * p  # 45
    
    # Initialize matrices using lists of lists
    U = [[0] * rank for _ in range(n * m)]  # 9 x 45
    V = [[0] * rank for _ in range(m * p)]  # 15 x 45
    W = [[0] * rank for _ in range(p * n)]  # 15 x 45
    
    col_idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                row_u = i * m + j
                row_v = j * p + k
                row_w = k * n + i
                
                U[row_u][col_idx] = 1
                V[row_v][col_idx] = 1
                W[row_w][col_idx] = 1
                col_idx += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
