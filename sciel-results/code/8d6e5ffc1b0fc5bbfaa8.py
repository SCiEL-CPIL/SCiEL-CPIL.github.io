import numpy as np
import time

N = 100

def compute_value(h):
    """Compute the objective value: max correlation / N * 2"""
    corr = np.correlate(h, 1.0 - h, mode="full")
    return np.max(corr) / N * 2

def local_search(h, max_iter=500, time_limit=2.0):
    """
    Perform greedy local search by swapping any pair of distinct elements
    to reduce max correlation.
    """
    start_time = time.time()
    current_val = compute_value(h)
    
    for iteration in range(max_iter):
        if time.time() - start_time > time_limit:
            break
        
        best_swap = None
        best_val = current_val
        improved = False
        
        # Try all pairs
        for i in range(N):
            for j in range(i + 1, N):
                if h[i] == h[j]:
                    continue
                
                # Perform swap
                h[i], h[j] = h[j], h[i]
                
                # Compute new value
                new_val = compute_value(h)
                
                # Undo swap
                h[i], h[j] = h[j], h[i]
                
                if new_val < best_val - 1e-12:
                    best_val = new_val
                    best_swap = (i, j)
                    improved = True
        
        if not improved:
            break
        
        # Apply best swap
        i, j = best_swap
        h[i], h[j] = h[j], h[i]
        current_val = best_val
    
    return h, current_val

def construct():
    # Define candidate multisets that sum to 50 (N/2)
    # 1. Binary: 50 zeros, 50 ones
    # 2. Ternary: 33 zeros, 34 halfes, 33 ones
    # 3. Quinary: 20 each of 0, 0.25, 0.5, 0.75, 1.0
    # 4. Hexary (approx): 17 of 0, 17 of 1/5, 17 of 2/5, 17 of 3/5, 17 of 4/5, 17 of 1.0?
    #    Sum = 17 * (0 + 0.2 + 0.4 + 0.6 + 0.8 + 1.0) = 17 * 3.0 = 51. Too high.
    #    Adjust counts: 16 of 0, 17 of 0.2, 17 of 0.4, 17 of 0.6, 17 of 0.8, 16 of 1.0
    #    Sum = 16*0 + 17*0.2 + 17*0.4 + 17*0.6 + 17*0.8 + 16*1.0
    #    = 3.4 + 6.8 + 10.2 + 13.6 + 16 = 50.0. Correct.
    
    candidates = [
        ("binary", np.array([0.0, 1.0]), [50, 50]),
        ("ternary", np.array([0.0, 0.5, 1.0]), [33, 34, 33]),
        ("quinary", np.array([0.0, 0.25, 0.5, 0.75, 1.0]), [20, 20, 20, 20, 20]),
        ("hexary", np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0]), [16, 17, 17, 17, 17, 16])
    ]
    
    best_h = None
    best_val = float('inf')
    
    np.random.seed(42)
    
    # Allocate time roughly equally, but allow more restarts for better candidates
    # Total time budget ~100s.
    # Let's try 10 restarts per candidate first to see which is best, then polish the best one?
    # Or just run a fixed number of restarts for all and pick the global best.
    # 4 candidates * 20 restarts * 2.0s = 160s. Too long.
    # 4 candidates * 10 restarts * 1.5s = 60s. Safe.
    
    num_restarts_per_candidate = 10
    time_limit_per_restart = 1.5
    
    for name, values, counts in candidates:
        base_h = np.concatenate([np.full(c, v) for v, c in zip(values, counts)])
        
        for restart in range(num_restarts_per_candidate):
            h = base_h.copy()
            np.random.shuffle(h)
            
            h_optimized, val = local_search(h, max_iter=500, time_limit=time_limit_per_restart)
            
            if val < best_val:
                best_val = val
                best_h = h_optimized.copy()
    
    # If we have time left, polish the best solution further
    if best_h is not None:
        remaining_time = 110.0 - (time.time() - start_construct_time) if 'start_construct_time' in locals() else 20.0
        # We didn't record start time, so let's just assume we have ~20s left if we were efficient.
        # Actually, let's just do a few more high-quality iterations on the best found.
        # To be safe and deterministic, we'll just return the best found so far.
        # The local search already ran for 1.5s per restart.
        
    return best_h
