"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-minimal (or best-known) exact tensor decomposition
    for multiplying a 3x5 matrix by a 5x7 matrix.
    
    Returns dict with keys 'U', 'V', 'W' where:
      U: shape (n*m, rank)
      V: shape (m*p, rank)  
      W: shape (p*n, rank), indexed as W[k*n + i, r]
    
    The decomposition satisfies:
      For all (i,j,k): sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
      And the full tensor equality holds.
    """
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # 10s safety margin before 120s limit
    
    # Schoolbook decomposition: rank = n*m*p
    # Each (i,j,k) triple corresponds to one rank-1 term
    rank = n * m * p
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
