import numpy as np
import time

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 110  # 10s margin before 120s limit
    
    n, m, p = 3, 4, 4
    rank = n * m * p
    
    # Ensure correct shapes as per error message: n*m, m*p, p*n
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    # Schoolbook decomposition
    t = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Check deadline
                if time.time() > deadline:
                    break
                # A index: i * m + j
                U[i * m + j, t] = 1.0
                # B index: j * p + k
                V[j * p + k, t] = 1.0
                # W index: k * n + i (as per prompt)
                W[k * n + i, t] = 1.0
                t += 1
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break
            
    return {"U": U, "V": V, "W": W}
