import time
import numpy as np

DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs an exact tensor decomposition for 2x3 @ 3x9 matrix multiplication.
    Uses the standard (schoolbook) decomposition which is exact and uses integer entries.
    Rank = 2 * 3 * 9 = 54.
    """
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Initialize matrices with zeros
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Map each (i, j, k) triplet to a unique rank index r
    # i in [0, n), j in [0, m), k in [0, p)
    # U index: i*m + j
    # V index: j*p + k
    # W index: k*n + i
    
    r = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Set the corresponding entries to 1
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                
                U[u_idx][r] = 1
                V[v_idx][r] = 1
                W[w_idx][r] = 1
                r += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
