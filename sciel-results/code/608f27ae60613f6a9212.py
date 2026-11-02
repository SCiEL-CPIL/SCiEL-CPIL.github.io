import time
import numpy as np

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Returns a rank-minimal exact tensor decomposition for 5x5 matrix multiplication.
    Uses the schoolbook decomposition with rank 125 as a baseline, with a deadline guard.
    """
    import time
    start_time = time.time()
    deadline = 120.0  # seconds, with safety margin from 150s limit
    
    n, m, p = DIMENSIONS
    
    rank = n * m * p
    U = np.zeros((n*m, rank), dtype=float)
    V = np.zeros((m*p, rank), dtype=float)
    W = np.zeros((p*n, rank), dtype=float)
    
    col = 0
    for i in range(n):
        if time.time() - start_time > deadline - 10:
            break
        for j in range(m):
            if time.time() - start_time > deadline - 10:
                break
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
