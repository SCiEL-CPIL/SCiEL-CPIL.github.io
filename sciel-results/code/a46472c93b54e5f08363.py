"""Matrix multiplication 3x3 x 3x8 rank-minimal exact decomposition."""

import numpy as np

DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Initialize matrices
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
                
    # Convert to lists of lists as per interface requirement?
    # The interface says: factor matrices of shapes ... with integer/half-integer entries.
    # It doesn't explicitly say they must be lists, but the seed returns lists.
    # "Keep the function names and signatures ... return interface EXACTLY as in the current program"
    # Current program returns {"U": U, "V": V, "W": W} where U,V,W are lists of lists.
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
