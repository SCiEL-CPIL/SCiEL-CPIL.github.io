"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
def construct():
    """
    Returns an exact tensor decomposition for multiplying a 2x3 matrix by a 3x9 matrix.
    Uses the trivial (coordinate) decomposition which is exact.
    """
    n, m, p = DIMENSIONS
    
    # Trivial decomposition rank is n*m*p
    rank = n * m * p
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Map each (i, j, k) to a unique rank index c
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # U index: i*m + j
                U[i * m + j][c] = 1
                # V index: j*p + k
                V[j * p + k][c] = 1
                # W index: k*n + i
                W[k * n + i][c] = 1
                c += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
