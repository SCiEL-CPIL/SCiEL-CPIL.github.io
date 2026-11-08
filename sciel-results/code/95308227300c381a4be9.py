import numpy as np
import time

DIMENSIONS = [6, 6, 6]

def construct():
    start_time = time.time()
    deadline = 120.0 # seconds
    
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Initialize arrays with zeros
    # Using float64 to be safe with "integer/half-integer" (integers are valid floats)
    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                
                U[u_idx, column] = 1.0
                V[v_idx, column] = 1.0
                W[w_idx, column] = 1.0
                column += 1
                
                # Safety check (not really needed for this small size, but good for directive)
                if time.time() > deadline - 5.0:
                    break
            if time.time() > deadline - 5.0:
                break
        if time.time() > deadline - 5.0:
            break
            
    return {"U": U, "V": V, "W": W}
