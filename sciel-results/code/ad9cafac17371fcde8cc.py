import numpy as np
import time
import random

DIMENSIONS = [4, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110.0
    
    # Part 2: 4x4 x 4x1 (Rank 4)
    U2 = np.zeros((n*m, 4))
    V2 = np.zeros((m*p, 4))
    W2 = np.zeros((p*n, 4))
    for j in range(m):
        r = j
        for i in range(n):
            U2[i*m + j, r] = 1.0
            V2[j*p + 4, r] = 1.0
            W2[4*n + i, r] = 1.0
            
    # Part 1: 4x4 x 4x4
    # Try to find rank R decomposition
    best_rank = 64
    best_U1 = None
    best_V1 = None
    best_W1 = None
    
    # Schoolbook fallback
    U1_fb = np.zeros((n*m, 64))
    V1_fb = np.zeros((m*4, 64))
    W1_fb = np.zeros((4*n, 64))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(4):
                U1_fb[i*m + j, col] = 1.0
                V1_fb[j*4 + k, col] = 1.0
                W1_fb[k*n + i, col] = 1.0
                col += 1
                
    best_U1 = U1_fb
    best_V1 = V1_fb
    best_W1 = W1_fb
    
    # Search for lower rank
    for R in range(19, 25):
        if time.time() > deadline - 20:
            break
            
        found = False
        for restart in range(10):
            if time.time() > deadline - 20:
                break
                
            U1 = np.random.randn(n*m, R)
            V1 = np.random.randn(m*4, R)
            W1 = np.random.randn(4*n, R)
            
            # Optimize
            for step in range(200):
                # Compute T1
                # T1[i,j,k] = sum_r U1[i*m+j, r] V1[j*4+k, r] W1[k*n+i, r]
                # We want T1[i,j,k] = 1
                
                # Calculate gradient? Too slow.
                # Just check error.
                # If error < 1e-6, we are good.
                
                # Sample some points
                err = 0
                for i in range(n):
                    for j in range(m):
                        for k in range(4):
                            val = 0
                            for r in range(R):
                                val += U1[i*m+j, r] * V1[j*4+k, r] * W1[k*n+i, r]
                            err += (val - 1)**2
                            
                if err < 1e-6:
                    found = True
                    break
                    
                # Simple update: random jitter
                U1 += np.random.randn(n*m, R) * 0.1
                V1 += np.random.randn(m*4, R) * 0.1
                W1 += np.random.randn(4*n, R) * 0.1
                
            if found:
                best_rank = R
                best_U1 = U1
                best_V1 = V1
                best_W1 = W1
                break
                
        if found:
            break
            
    # Combine
    U = np.hstack([best_U1, U2])
    V = np.hstack([best_V1, V2])
    W = np.hstack([best_W1, W2])
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
