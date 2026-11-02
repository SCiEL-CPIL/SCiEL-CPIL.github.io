import time
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    """
    Returns a rank-27 exact tensor decomposition for 3x3 matrix multiplication.
    This is the schoolbook decomposition, which is guaranteed to be correct.
    """
    n, m, p = 3, 3, 3
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
                W[k * n + i][col] = 1
                col += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
