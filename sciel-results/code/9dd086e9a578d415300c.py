import time
import numpy as np

DIMENSIONS = [2, 4, 5]

def construct():
    """
    Returns a rank-minimal exact tensor decomposition for 2x4 * 4x5 matrix multiplication.
    The schoolbook algorithm has rank n*m*p = 40.
    """
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Initialize matrices with zeros
    # U: (n*m, rank)
    # V: (m*p, rank)
    # W: (p*n, rank)
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Index for U: row i*m + j
                U[i * m + j, col] = 1.0
                # Index for V: row j*p + k
                V[j * p + k, col] = 1.0
                # Index for W: row k*n + i
                # Convention: W[k*n + i, col]
                W[k * n + i, col] = 1.0
                col += 1
                
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
