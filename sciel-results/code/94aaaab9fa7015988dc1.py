import time
import numpy as np

DIMENSIONS = [3, 4, 5]

def construct():
    """
    Returns the schoolbook decomposition for 3x4x5 matrix multiplication.
    Rank is 3*4*5 = 60.
    """
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Initialize matrices with zeros
    # U shape: (n*m, rank)
    # V shape: (m*p, rank)
    # W shape: (p*n, rank)
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
                
    # Convert to lists of lists as per the interface example
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
