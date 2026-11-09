import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [3, 4, 6]

def _schoolbook_decomposition(n, m, p):
    """Return the schoolbook decomposition of rank n*m*p."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                U[u_idx][col] = 1
                V[v_idx][col] = 1
                W[w_idx][col] = 1
                col += 1
    
    return {"U": U, "V": V, "W": W}

def construct():
    """
    Returns a rank-minimal exact tensor decomposition for matrix multiplication 3x4 by 4x6.
    Currently returns the schoolbook decomposition of rank 72.
    """
    n, m, p = DIMENSIONS
    result = _schoolbook_decomposition(n, m, p)
    return result
