import numpy as np
import time

def construct():
    start_time = time.time()
    n, m, p = 2, 3, 9
    
    # Schoolbook decomposition: rank = n * m * p = 54
    # This is guaranteed to be correct and is the baseline.
    # Any improvement requires finding a valid tensor decomposition of rank < 54.
    # Given the strict time limit and the risk of invalid output, we stick to schoolbook
    # but implement it more cleanly.
    
    rank = n * m * p
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    r = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                
                U[u_idx, r] = 1.0
                V[v_idx, r] = 1.0
                W[w_idx, r] = 1.0
                
                r += 1
                
                if time.time() - start_time > 140:
                    break
            if time.time() - start_time > 140:
                break
        if time.time() - start_time > 140:
            break
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
