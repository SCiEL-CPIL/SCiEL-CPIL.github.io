"""Matrix multiplication 2x4x4: rank-minimal exact tensor decomposition."""
import numpy as np
import time
import itertools

DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 2, 4, 4
    deadline = time.time() + 140.0
    
    # We'll try to find a low-rank decomposition.
    # The tensor T has indices (i, j, k) with i in [0,n), j in [0,m), k in [0,p).
    # T[i,j,k] = 1 for all (i,j,k).
    # We need U (n*m x r), V (m*p x r), W (p*n x r) such that
    # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all (i,j,k).
    
    # Strategy: Start with rank 16 (schoolbook) and try to reduce.
    # Actually, let's think about the structure.
    # T[i,j,k] = 1. This is the all-ones tensor.
    # The rank of the all-ones tensor of size n x m x p is 1!
    # Wait, is that right?
    # If U[i*m+j, 0] = 1, V[j*p+k, 0] = 1, W[k*n+i, 0] = 1,
    # then sum_c U[...] V[...] W[...] = 1*1*1 = 1.
    # So rank 1 works!
    
    # Let me verify:
    # U is (n*m) x 1, all entries 1.
    # V is (m*p) x 1, all entries 1.
    # W is (p*n) x 1, all entries 1.
    # For any (i,j,k): U[i*m+j, 0] * V[j*p+k, 0] * W[k*n+i, 0] = 1*1*1 = 1.
    # Yes! Rank 1 is sufficient.
    
    # So the minimal rank is 1.
    rank = 1
    U = [[1] for _ in range(n * m)]
    V = [[1] for _ in range(m * p)]
    W = [[1] for _ in range(p * n)]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
