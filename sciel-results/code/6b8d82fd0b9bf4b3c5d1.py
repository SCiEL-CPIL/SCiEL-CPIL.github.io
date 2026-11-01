import numpy as np
import random
import time

def construct():
    n, m, p = 2, 2, 5
    # Target rank
    target_rank = 18
    
    # Deadline guard
    start_time = time.time()
    timeout = 100  # Leave 20s margin
    
    # Trivial rank 20 solution as fallback
    # U (4, 20), V (10, 20), W (10, 20)
    U_fallback = np.zeros((n*m, n*m*p))
    V_fallback = np.zeros((m*p, n*m*p))
    W_fallback = np.zeros((p*n, n*m*p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                t = i*m*p + j*p + k
                U_fallback[i*m+j, t] = 1.0
                V_fallback[j*p+k, t] = 1.0
                W_fallback[k*n+i, t] = 1.0
                
    best_U, best_V, best_W = U_fallback, V_fallback, W_fallback
    best_rank = n*m*p
    
    # Try to find lower rank
    for rank in range(target_rank, 15, -1):
        if time.time() - start_time > timeout:
            break
            
        for attempt in range(50):
            if time.time() - start_time > timeout:
                break
                
            # Random initialization
            U = np.random.randn(n*m, rank)
            V = np.random.randn(m*p, rank)
            W = np.random.randn(p*n, rank)
            
            # Check validity
            valid = True
            # Check all triples
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        u_idx = i*m + j
                        v_idx = j*p + k
                        w_idx = k*n + i
                        s = np.sum(U[u_idx] * V[v_idx] * W[w_idx])
                        if abs(s - 1.0) > 1e-5:
                            valid = False
                            break
                        # Check incompatible (sample a few)
                        if j < m-1:
                            j2 = j + 1
                            v_idx2 = j2*p + k
                            s2 = np.sum(U[u_idx] * V[v_idx2] * W[w_idx])
                            if abs(s2) > 1e-5:
                                valid = False
                                break
                    if not valid: break
                if not valid: break
            if not valid: break
            
            if valid:
                best_U, best_V, best_W = U, V, W
                best_rank = rank
                break # Found solution for this rank, try lower? No, break inner loop, continue outer to try lower rank?
                # Actually, if we find rank 18, we can stop if we don't want to risk timeout for 17.
                # But let's try lower.
                
    return {'U': best_U, 'V': best_V, 'W': best_W}
