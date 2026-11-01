"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [2, 2, 2]

# EVOLVE-BLOCK-START
def construct():
    """
    Return a rank-7 tensor decomposition for 2x2 matrix multiplication.
    
    The decomposition is based on Strassen's algorithm, which is provably
    optimal for 2x2 matrices (lower bound of 7 established by Landsberg).
    
    Interface:
    - U: shape (n*m, rank) = (4, 7), row index = i*m + j
    - V: shape (m*p, rank) = (4, 7), row index = j*p + k  
    - W: shape (p*n, rank) = (4, 7), row index = k*n + i
    
    The decomposition satisfies:
    sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k
    
    This represents the bilinear map C[i,k] = sum_j A[i,j] * B[j,k].
    """
    n, m, p = DIMENSIONS
    rank = 7
    
    # Strassen's rank-7 decomposition coefficients
    # These are integer entries that exactly decompose the 2x2 matmul tensor
    
    U_data = (
        (1, 0, 1, 0, 1, -1, 0),   # i=0, j=0
        (0, 0, 0, 0, 1, 0, 1),   # i=0, j=1
        (0, 1, 0, 0, 0, 1, 0),   # i=1, j=0
        (1, 1, 0, 1, 0, 0, -1)   # i=1, j=1
    )
    
    V_data = (
        (1, 1, 0, -1, 0, 1, 0),  # j=0, k=0
        (0, 0, 1, 0, 0, 1, 0),   # j=0, k=1
        (0, 0, 0, 1, 0, 0, 1),   # j=1, k=0
        (1, 0, -1, 0, 1, 0, 1)   # j=1, k=1
    )
    
    W_data = (
        (1, 0, 0, 1, -1, 0, 1),  # k=0, i=0
        (0, 1, 0, 1, 0, 0, 0),   # k=0, i=1
        (0, 0, 1, 0, 1, 0, 0),   # k=1, i=0
        (1, -1, 1, 0, 0, 1, 0)   # k=1, i=1
    )
    
    # Convert to lists as required by interface
    U = [list(row) for row in U_data]
    V = [list(row) for row in V_data]
    W = [list(row) for row in W_data]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
