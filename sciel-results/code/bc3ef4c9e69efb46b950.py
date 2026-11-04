"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-minimal exact tensor decomposition for multiplying
    a 3x3 matrix by a 3x6 matrix.
    
    Current strategy: Schoolbook multiplication.
    Rank: 3 * 3 * 6 = 54.
    
    This is the baseline. Improvements would require finding a decomposition
    with rank < 54, e.g., by using Strassen-like algorithms for sub-blocks.
    For 3x3x3, the best known rank is 23, so 3x3x6 could be 46.
    However, implementing a specific rank-23 decomposition is complex.
    We return the robust schoolbook decomposition.
    """
    n, m, p = DIMENSIONS
    
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
