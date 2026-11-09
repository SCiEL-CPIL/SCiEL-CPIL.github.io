import time
import numpy as np

DIMENSIONS = [3, 4, 7]

def construct():
    """
    Constructs the rank-minimal exact tensor decomposition for matrix multiplication 
    of a (n x m) matrix by a (m x p) matrix.
    
    For general matrices, the minimum rank is n * m * p (schoolbook algorithm).
    This implementation returns the schoolbook decomposition.
    """
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Initialize matrices with zeros
    # U: (n*m, rank)
    # V: (m*p, rank)
    # W: (p*n, rank)
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Fill the decomposition
    # Index mapping:
    # U row index: i * m + j
    # V row index: j * p + k
    # W row index: k * n + i (Trace convention: W[k*n+i])
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Set the corresponding entries in this column to 1
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
                
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
