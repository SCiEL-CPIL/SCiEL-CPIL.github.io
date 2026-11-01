"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 9]

import time
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # The multiplication tensor for A (n x m) and B (m x p) producing C (n x p)
    # is defined by T[i, j, k] = 1 if j == k and j < m, else 0.
    # This tensor is separable as u_i * M_{j,k} where u is a vector of ones 
    # of length n and M is the m x p matrix with 1s on the first m diagonal 
    # entries and 0s elsewhere.
    # The rank of M is m. Thus the rank of the tensor is m.
    
    rank = m
    
    # U shape: (n*m, rank)
    # V shape: (m*p, rank)
    # W shape: (p*n, rank)
    
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    # U[i, j, c] = 1 for all i, j, c
    U[:] = 1
    
    # V[j, k, c] = 1 if j == c else 0
    # V is flattened as [j*p + k, c]
    for j in range(m):
        for c in range(rank):
            if j == c:
                # Set all k for this j and c to 1
                V[j*p : (j+1)*p, c] = 1
            # else remains 0
            
    # W[k, i, c] = 1 if k == c else 0
    # W is flattened as [k*n + i, c]
    for k in range(p):
        for c in range(rank):
            if k == c:
                # Set all i for this k and c to 1
                W[k*n : (k+1)*n, c] = 1
            # else remains 0

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
