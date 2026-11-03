# EVOLVE-BLOCK-START
import time
import numpy as np

DIMENSIONS = [5, 5, 5]

def _build_schoolbook():
    """
    Construct the schoolbook decomposition for 5x5 matrix multiplication.
    Returns a dictionary with keys 'U', 'V', 'W'.
    """
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Using numpy for efficient initialization and assignment
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
                
    return {"U": U, "V": V, "W": W}

def construct():
    """
    Main entry point. Returns the factor matrices.
    """
    return _build_schoolbook()
# EVOLVE-BLOCK-END
