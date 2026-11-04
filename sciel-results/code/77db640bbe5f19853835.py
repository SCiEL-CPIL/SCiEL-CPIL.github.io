import numpy as np

DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START
import time
_start_time = time.time()
_TIMEOUT = 150.0
_SAFETY_MARGIN = 5.0

def _time_remaining():
    return _TIMEOUT - _SAFETY_MARGIN - (time.time() - _start_time)

def construct():
    """
    Constructs the factor matrices for the matrix multiplication tensor 
    of a 3x3 matrix multiplied by a 3x8 matrix.
    
    Uses the schoolbook (naive) bilinear decomposition with rank 72.
    """
    n, m, p = DIMENSIONS
    rank = n * m * p  # 3*3*8 = 72
    
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                
                U[u_idx, col] = 1
                V[v_idx, col] = 1
                W[w_idx, col] = 1
                col += 1
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
# EVOLVE-BLOCK-END
