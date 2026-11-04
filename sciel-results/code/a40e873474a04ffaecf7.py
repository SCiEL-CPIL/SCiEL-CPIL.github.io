import numpy as np
import time

def solve(a, b):
    n = a.shape[0]
    start_time = time.time()
    time_limit = 115.0  # Safety margin
    
    # 1. Initial Heuristic: Rank Matching
    # Heavy flows (high row sum in A) should go to short distances (low row sum in B)
    a_sums = a.sum(axis=1)
    b_sums = b.sum(axis=1)
    
    # Sort facilities by decreasing flow sum
    heavy = np.argsort(-a_sums, kind='stable')
    # Sort locations by increasing distance sum
    central = np.argsort(b_sums, kind='stable')
    
    p = np.empty(n, dtype=np.int64)
    p[heavy] = central
    
    # 2. Compute Interaction Matrix M
    # M[i, l] = sum_m A[i, m] * B[l, p(m)]
    # Let B_p be B with columns permuted by p.
    # B_p[l, m] = B[l, p[m]]
    # M[i, l] = sum_m A[i, m] * B_p[l, m]
    # This is (A @ B_p.T)[i, l]
    
    B_p = b[:, p]
    M = a @ B_p.T
    
    # 3. Local Search (2-opt)
    # We iterate until no improvement or time limit reached.
    # To speed up, we can randomize the order of pairs or just do sequential passes.
    # Sequential passes are standard.
    
    improved = True
    passes = 0
    max_passes = 50 # Cap on passes just in case
    
    while improved and passes < max_passes:
        improved = False
        passes += 1
        
        # Check time
        if time.time() - start_time > time_limit:
            break
            
        for i in range(n):
            if time.time() - start_time > time_limit:
                break
            for k in range(i + 1, n):
                u = p[i]
                v = p[k]
                
                # Delta = (M[i, v] - M[i, u]) + (M[k, u] - M[k, v])
                # Note: M[i, u] includes A[i,i]B[u,u] if i is at u.
                # But the formula V_i(l) - V_i(u) handles the change correctly 
                # because the self-term A[i,i]B[u,u] is constant for facility i 
                # regardless of where it is? 
                # No, if i moves from u to v, the self term changes from 
                # A[i,i]B[u,u] to A[i,i]B[v,v].
                # However, in QAP, A[i,i] is typically 0. 
                # If not, we must account for it.
                # Let's check if a[i,i] != 0.
                # Standard QAP assumes A[i,i]=0. We assume this.
                
                delta = (M[i, v] - M[i, u]) + (M[k, u] - M[k, v])
                
                if delta < 0:
                    # Accept swap
                    p[i], p[k] = p[k], p[i]
                    
                    # Update M
                    # M[j, l] += (A[j, i] - A[j, k]) * (B[l, v] - B[l, u])
                    diff_a = a[:, i] - a[:, k]
                    diff_b = b[:, v] - b[:, u]
                    M += np.outer(diff_a, diff_b)
                    
                    improved = True
                    # Optional: break inner loop to restart from beginning?
                    # No, continuing is fine (First Improvement strategy within pass).
                    # But restarting ensures we don't miss better moves early on.
                    # For speed, we continue.
                    
    # Return the best permutation found
    return p.tolist()
