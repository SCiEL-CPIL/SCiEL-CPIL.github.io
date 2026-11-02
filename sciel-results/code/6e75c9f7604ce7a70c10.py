import numpy as np
import time

DIMENSIONS = [3, 3, 6]

def construct():
    n, m, p = DIMENSIONS
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 120 - 10  # 110s safety margin
    
    # Schoolbook decomposition: Rank R = n * m * p
    R = n * m * p
    
    U = np.zeros((n * m, R))
    V = np.zeros((m * p, R))
    W = np.zeros((p * n, R))
    
    r = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                
                U[a, r] = 1.0
                V[b, r] = 1.0
                W[c, r] = 1.0
                r += 1
                
    # Verification (optional, but good for debugging)
    # For this simple case, it's guaranteed to be correct.
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
