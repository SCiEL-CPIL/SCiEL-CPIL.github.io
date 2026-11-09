"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 7]

# EVOLVE-BLOCK-START
import numpy as np
import random

def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    total_eq = n * m * p
    
    # Schoolbook decomposition
    schoolbook_rank = total_eq
    U = np.zeros((nm, schoolbook_rank))
    V = np.zeros((mp, schoolbook_rank))
    W = np.zeros((pn, schoolbook_rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    
    # Try to reduce rank using randomized rank reduction
    # We'll use a simple approach: try to find a lower rank decomposition
    # by using random linear combinations and checking if they still work
    
    # Actually, let's try a known result: for small matrices, sometimes
    # the rank can be reduced. But for 3x4 * 4x7, the schoolbook is 84.
    # Let's just return the schoolbook for now, as finding a lower rank
    # decomposition is a hard open problem in general.
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
