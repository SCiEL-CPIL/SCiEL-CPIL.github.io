"""Tensor decomposition search for 4x5x5 matrix multiplication."""
import time
import numpy as np

DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Returns a rank-minimal (or best-effort) exact tensor decomposition
    for multiplying a 4x5 matrix by a 5x5 matrix.
    
    Returns: {'U': (20, rank), 'V': (25, rank), 'W': (20, rank)}
    where the tensor T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
    equals 1 if the indices are valid, 0 otherwise.
    
    Convention: W[k*n+i, r] where n=4, m=5, p=5.
    """
    n, m, p = DIMENSIONS
    
    # Schoolbook decomposition (rank = 100)
    rank_schoolbook = n * m * p
    U = np.zeros((n * m, rank_schoolbook))
    V = np.zeros((m * p, rank_schoolbook))
    W = np.zeros((p * n, rank_schoolbook))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
# EVOLVE-BLOCK-END
