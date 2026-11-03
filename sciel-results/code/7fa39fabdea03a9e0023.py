"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 9]

import time
import random
import numpy as np
from itertools import product

# EVOLVE-BLOCK-START
import time

def _schoolbook_decomp(n, m, p):
    """
    Returns schoolbook (trivial) tensor decomposition for n x m * m x p matrix multiplication.
    Rank = n * m * p.
    U: shape (n*m, rank)
    V: shape (m*p, rank)
    W: shape (p*n, rank)
    
    For each (i, j, l) triple, one rank-1 term:
      U[i*m + j, k] = 1
      V[j*p + l, k] = 1
      W[l*n + i, k] = 1
    where k = i*(m*p) + j*p + l
    """
    rank = n * m * p
    dim_U = n * m
    dim_V = m * p
    dim_W = p * n
    
    U = [[0] * rank for _ in range(dim_U)]
    V = [[0] * rank for _ in range(dim_V)]
    W = [[0] * rank for _ in range(dim_W)]
    
    for i in range(n):
        for j in range(m):
            for l in range(p):
                k = i * (m * p) + j * p + l
                U[i * m + j][k] = 1
                V[j * p + l][k] = 1
                W[l * n + i][k] = 1
    
    return U, V, W

def construct():
    """
    Returns a tensor decomposition for 4x4 * 4x9 matrix multiplication.
    Uses schoolbook decomposition with rank 144.
    """
    start_time = time.time()
    deadline = start_time + 120.0
    
    n, m, p = DIMENSIONS
    
    # Check time before starting
    if time.time() > deadline - 5:
        # Fallback: return empty (shouldn't happen)
        return {"U": [], "V": [], "W": []}
    
    U, V, W = _schoolbook_decomp(n, m, p)
    
    # Verify the decomposition is correct
    # The tensor identity: sum_k U[i*m+j,k] * V[j*p+l,k] * W[l*n+i,k] should equal 1 for all i,j,l
    # In schoolbook, for each (i,j,l), there's exactly one k where all three are 1, so sum = 1. Correct.
    
    if time.time() > deadline - 2:
        return {"U": U, "V": V, "W": W}
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
