"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
import time
import sys
import numpy as np

def construct():
    """
    Returns a rank-minimal exact tensor decomposition for multiplying
    a 3x3 matrix by a 3x5 matrix.
    
    Uses schoolbook decomposition with rank n*m*p = 45.
    This is the standard exact decomposition for matrix multiplication.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Build schoolbook decomposition
    U = np.zeros((n * m, rank), dtype=np.int32)
    V = np.zeros((m * p, rank), dtype=np.int32)
    W = np.zeros((p * n, rank), dtype=np.int32)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    
    # Convert to lists for the required interface
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
