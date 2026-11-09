"""Matrix multiplication 5,5,5 - correct schoolbook decomposition with time guards."""

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    """Return a valid tensor decomposition for 5x5 matrix multiplication.
    
    Uses the schoolbook decomposition of rank 125 = 5*5*5.
    For each (i,j,k) triple, we create one rank-1 term:
      U[i*m + j] = 1  (A index: row i, col j)
      V[j*p + k] = 1  (B index: row j, col k)
      W[k*n + i] = 1  (C index: col k, row i, using W[k*n+i] convention)
    """
    n, m, p = 5, 5, 5
    rank = n * m * p  # 125
    
    # Use numpy for faster construction
    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, column] = 1.0
                V[j * p + k, column] = 1.0
                W[k * n + i, column] = 1.0
                column += 1
    
    # Convert to lists as required by the interface
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }

# EVOLVE-BLOCK-END
