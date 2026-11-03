import numpy as np
import time
import random

DIMENSIONS = [4, 4, 6]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 110
    
    # Fallback
    rank_school = n * m * p
    U_school = np.zeros((n*m, rank_school))
    V_school = np.zeros((m*p, rank_school))
    W_school = np.zeros((p*n, rank_school))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx = i*m*p + j*p + k
                U_school[i*m+j, idx] = 1
                V_school[j*p+k, idx] = 1
                W_school[k*n+i, idx] = 1
                
    fallback = {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}
    
    # Try rank 32
    R = 32
    if time.time() < deadline:
        # Initialize
        U = np.random.randn(n*m, R)
        V = np.random.randn(m*p, R)
        W = np.random.randn(p*n, R)
        
        # ALS
        for it in range(100):
            if time.time() > deadline: break
            # Update U
            for r in range(R):
                # V[:, r] and W[:, r] are fixed
                # Minimize sum (T - sum U V W)^2
                # This is complex.
                pass
        
    return fallback
