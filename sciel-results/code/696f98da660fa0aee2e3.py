"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for matrix multiplication 3x3 * 3x7.
    
    The rank of the (n,m,p) matrix multiplication tensor is the minimum number
    of rank-1 tensors whose sum equals the multiplication tensor. For (3,3,7),
    the 7 output columns are independent, and each 3x3 x 3x1 multiplication
    requires rank 9. Thus the total rank is 7*9 = 63, which is achieved by
    the schoolbook algorithm.
    
    Returns dict with 'U', 'V', 'W' as lists of lists (integer entries).
    """
    n, m, p = DIMENSIONS
    
    # Schoolbook solution: rank = n * m * p = 3 * 3 * 7 = 63
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
