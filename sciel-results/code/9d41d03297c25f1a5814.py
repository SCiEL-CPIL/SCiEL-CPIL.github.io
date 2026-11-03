import numpy as np
import time

def solve(a, b):
    n = a.shape[0]
    start_time = time.time()
    time_limit = 115.0  # Safety margin
    
    # Initial solution: Rank matching
    heavy = np.argsort(-a.sum(axis=1), kind="stable")
    central = np.argsort(b.sum(axis=1), kind="stable")
    p = np.empty(n, dtype=np.int64)
    p[heavy] = central
    
    # Precompute some things? No, just use a and b directly.
    
    # Local Search 2-opt
    improved = True
    while improved:
        improved = False
        # Check time
        if time.time() - start_time > time_limit:
            break
            
        # We can try to find the best move or first improving move.
        # First improving is usually faster per pass, but might need more passes.
        # Best improving is slower per pass but fewer passes.
        # Let's try first improving for speed, or a mix.
        
        # To make it robust, let's do a full scan and pick the best move if time allows, 
        # or just first improving.
        # Given n=256, a full scan is 32k pairs.
        # Let's do a full scan and pick the best delta.
        
        best_delta = 0
        best_u, best_v = -1, -1
        
        for u in range(n):
            for v in range(u + 1, n):
                lu = p[u]
                lv = p[v]
                
                # delta = sum_{k != u,v} (a[u,k] - a[v,k]) * (b[lv,k] - b[lu,k])
                
                # Vectorized calculation
                # a_diff = a[u] - a[v]
                # b_diff = b[lv] - b[lu]
                # delta = np.dot(a_diff, b_diff) - a_diff[u]*b_diff[u] - a_diff[v]*b_diff[v]
                
                # We can compute this more efficiently by precomputing row differences?
                # No, just do it directly.
                
                a_diff = a[u] - a[v]
                b_diff = b[lv] - b[lu]
                
                # Exclude u and v
                # Instead of creating a mask, we can subtract the terms for u and v
                delta = np.dot(a_diff, b_diff)
                delta -= a_diff[u] * b_diff[u]
                delta -= a_diff[v] * b_diff[v]
                
                if delta < best_delta:
                    best_delta = delta
                    best_u = u
                    best_v = v
                    
        if best_u != -1:
            # Apply the move
            p[best_u], p[best_v] = p[best_v], p[best_u]
            improved = True
            # Update cost? We don't strictly need the absolute cost for the search, 
            # only the deltas. But we need to return the permutation.
            
    return p.tolist()
