import numpy as np
import time
import random

def construct():
    start_time = time.time()
    DEADLINE = 110  # safety margin
    
    n, m, p = 2, 3, 6
    nm, mp, np_ = n*m, m*p, p*n
    
    # Build target tensor
    T = np.zeros((nm, mp, np_))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    
    best_rank = n * m * p
    best_result = None
    
    # Schoolbook as initial
    U0 = np.zeros((nm, best_rank))
    V0 = np.zeros((mp, best_rank))
    W0 = np.zeros((np_, best_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U0[i*m+j, col] = 1
                V0[j*p+k, col] = 1
                W0[k*n+i, col] = 1
                col += 1
    best_result = {"U": U0.tolist(), "V": V0.tolist(), "W": W0.tolist()}
    
    # Try lower ranks
    for rank in range(18, best_rank):
        if time.time() > DEADLINE:
            break
        
        for attempt in range(20):
            if time.time() > DEADLINE:
                break
            
            # Random initialization
            U = np.random.randn(nm, rank)
            V = np.random.randn(mp, rank)
            W = np.random.randn(np_, rank)
            
            # Iterative refinement (alternating least squares style)
            for iteration in range(100):
                if time.time() > DEADLINE:
                    break
                
                # Update U
                for a in range(nm):
                    # Minimize sum_{b,c} (T[a,b,c] - sum_r U[a,r]V[b,r]W[c,r])^2
                    # Gradient: sum_{b,c} (T[a,b,c] - pred) * V[b,:] * W[c,:]
                    pred = np.einsum('ar,br,cr->abc', U, V, W)
                    resid = T - pred
                    grad = np.einsum('abc,br,cr->ar', resid, V, W)
                    U[a] -= 0.01 * grad
                
                # Similar for V and W... this is getting expensive.
            
            # Check
            pred = np.einsum('ar,br,cr->abc', U, V, W)
            err = np.max(np.abs(pred - T))
            if err < 1e-6:
                best_rank = rank
                best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                break
        
        if best_rank < n*m*p:
            break
    
    return best_result
