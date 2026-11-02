"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import product

def construct():
    n, m, p = 2, 3, 6
    
    # Start with the naive rank-36 decomposition
    rank = n * m * p
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                r = i * m * p + j * p + k
                U[i * m + j, r] = 1
                V[j * p + k, r] = 1
                W[k * n + i, r] = 1
    
    # Try to reduce rank using structured elimination
    # The key insight: for 2x3x6, we can potentially use rank 22
    # by combining two 2x3x3 blocks with rank 11 each
    
    # First, let's try a simple approach: check if we can eliminate some terms
    # by finding linear dependencies
    
    # Convert to list format for return
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
