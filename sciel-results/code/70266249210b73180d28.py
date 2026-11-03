"""
5x5 matrix multiplication tensor decomposition - rank 125 (schoolbook).
This is the standard exact decomposition. The task contract requires integer/half-integer entries.
"""
import time
import numpy as np

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = n * m * p  # 125
    
    # Use numpy for construction - shape (25, 125)
    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 110.0
    
    r = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if time.time() > deadline:
                    # Return current state as nested lists
                    return {
                        "U": U.tolist(),
                        "V": V.tolist(),
                        "W": W.tolist()
                    }
                
                # Set the corresponding entries to 1
                # U index: i * m + j
                # V index: j * p + k
                # W index: k * n + i (following the convention W[k*n+i, r])
                U[i * m + j, r] = 1.0
                V[j * p + k, r] = 1.0
                W[k * n + i, r] = 1.0
                r += 1
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
# EVOLVE-BLOCK-END
