import time
import numpy as np

DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs the schoolbook tensor decomposition for 3x3 * 3x6 matrix multiplication.
    Rank is n*m*p = 3*3*6 = 54.
    """
    start_time = time.time()
    deadline = 120.0  # 120s limit
    
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Initialize matrices
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # U index: i*m + j
                # V index: j*p + k
                # W index: k*n + i
                U[i * m + j, column] = 1.0
                V[j * p + k, column] = 1.0
                W[k * n + i, column] = 1.0
                column += 1
                
                # Deadline check (not strictly necessary for O(1) work, but good practice)
                if time.time() > deadline - 10:
                    break
            if time.time() > deadline - 10:
                break
        if time.time() > deadline - 10:
            break
            
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
