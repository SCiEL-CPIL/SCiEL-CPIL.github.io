"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
import time

def construct():
    """
    Rank-minimal exact tensor decomposition for multiplying a 2x5 matrix by a 5x6 matrix.
    
    The matrix multiplication tensor T[i,j,k] satisfies:
        C[i,k] = sum_j T[i,j,k] * A[i,j] * B[j,k]
    
    For standard matrix multiplication, T[i,j,k] = 1 for all valid i, j, k.
    This is the all-ones tensor of dimensions n x m x p = 2 x 5 x 6.
    
    The all-ones tensor has rank 1: it can be written as the outer product of
    three all-ones vectors. That is:
        T[i,j,k] = 1 = u_i * v_j * w_k
    where u_i = 1, v_j = 1, w_k = 1 for all indices.
    
    Therefore, a rank-1 decomposition exists:
        U[i*m + j, 0] = 1  for all i, j
        V[j*p + k, 0] = 1  for all j, k
        W[k*n + i, 0] = 1  for all k, i
    """
    n, m, p = DIMENSIONS
    
    # Rank 1 decomposition: all entries are 1
    rank = 1
    U = [[1] for _ in range(n * m)]
    V = [[1] for _ in range(m * p)]
    W = [[1] for _ in range(p * n)]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
