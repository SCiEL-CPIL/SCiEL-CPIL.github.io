"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _schoolbook():
    """
    Returns the schoolbook decomposition for matrix multiplication.
    For multiplying an (n x m) matrix by an (m x p) matrix, the rank is n*m*p.
    
    The tensor T is defined such that T[i, j, k] = 1 for all valid indices,
    where:
    - i in [0, n)
    - j in [0, m)
    - k in [0, p)
    
    The factor matrices are:
    - U of shape (n*m, rank), where U[i*m + j, col] = 1
    - V of shape (m*p, rank), where V[j*p + k, col] = 1
    - W of shape (p*n, rank), where W[k*n + i, col] = 1
    
    The decomposition satisfies:
    sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i, j, k
    """
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Initialize with zeros
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Each (i, j, k) triple corresponds to one rank-1 term
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
                
    return {"U": U, "V": V, "W": W}

def construct():
    """
    Construct the decomposition.
    For 3x5x5, the schoolbook rank is 75.
    No known sub-schoolbook decomposition exists that is easily findable 
    within the time constraints. We return the schoolbook decomposition.
    """
    return _schoolbook()
# EVOLVE-BLOCK-END
