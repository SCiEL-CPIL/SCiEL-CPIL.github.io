import numpy as np
import time

DIMENSIONS = [2, 3, 9]

def construct():
    start_time = time.time()
    deadline = start_time + 145  # 5 second margin
    
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Initialize with zeros
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if time.time() > deadline:
                    # Return whatever we have so far (should be complete if fast enough)
                    # But if we are in the middle, it's invalid. 
                    # Given the loops are small (54 iterations), this won't happen.
                    break
                U[i * m + j, column] = 1
                V[j * p + k, column] = 1
                W[k * n + i, column] = 1
                column += 1
    
    return {"U": U, "V": V, "W": W}
