import time
import sys
import numpy as np
from fractions import Fraction

DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
def _schoolbook(n, m, p):
    """Return schoolbook decomposition with rank n*m*p."""
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=object)
    V = np.zeros((m * p, rank), dtype=object)
    W = np.zeros((p * n, rank), dtype=object)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    
    return U, V, W

def construct():
    """
    Returns a rank-minimal exact tensor decomposition for matrix multiplication.
    Currently returns the schoolbook decomposition which is guaranteed to be valid.
    """
    n, m, p = DIMENSIONS
    U, V, W = _schoolbook(n, m, p)
    
    # Convert to lists to ensure JSON serializability and interface compliance
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
