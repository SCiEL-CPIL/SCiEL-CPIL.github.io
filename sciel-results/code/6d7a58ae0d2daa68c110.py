"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.linalg import svd
import time
import random

DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START

def _schoolbook_decomposition(n, m, p):
    """Standard schoolbook decomposition of rank n*m*p."""
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=np.int64)
    V = np.zeros((m * p, rank), dtype=np.int64)
    W = np.zeros((p * n, rank), dtype=np.int64)
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, column] = 1
                V[j * p + k, column] = 1
                W[k * n + i, column] = 1
                column += 1
    
    return {'U': U, 'V': V, 'W': W, 'rank': rank}

def construct():
    n, m, p = DIMENSIONS
    sb = _schoolbook_decomposition(n, m, p)
    U = sb['U'].tolist()
    V = sb['V'].tolist()
    W = sb['W'].tolist()
    return {'U': U, 'V': V, 'W': W}

# EVOLVE-BLOCK-END
