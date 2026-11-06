import time
import random
import numpy as np
import scipy.sparse as sp

def construct():
    start_time = time.time()
    time_limit = 55.0 # Safety margin
    
    N = 256
    
    # 1. Precompute APs
    ap_indices = []
    for d in range(1, N // 2 + 1):
        for start in range(1, N - d + 1):
            indices = list(range(start - 1, N, d))
            if len(indices) >= 2: # Singletons are trivial, but let's include them? 
                # The prompt says "including one". 
                # Sum of singleton is +/- 1. 
                # If we include them, max is at least 1.
                # If we exclude them, we must ensure the result is at least 1.
                # It's easier to include them.
                ap_indices.append(indices)
                
    # Note: The number of APs is large. 
    # d=1: ~33k
    # d=2: ~16k
    # Total ~100k.
    
    # Build sparse matrix
    # Rows = APs, Cols = N
    row_idx = []
    col_idx = []
    data = []
    
    for r, indices in enumerate(ap_indices):
        for c in indices:
            row_idx.append(r)
            col_idx.append(c)
            data.append(1)
            
    A = sp.csr_matrix((data, (row_idx, col_idx)), shape=(len(ap_indices), N))
    
    # 2. Initialization
    # Start with a random sequence
    rng = random.Random(42)
    best_s = np.array([rng.choice([-1, 1]) for _ in range(N)])
    
    # Calculate initial score
    sums = A.dot(best_s)
    best_score = np.max(np.abs(sums))
    
    # 3. Local Search (Simulated Annealing)
    T = 1.0
    cooling = 0.999
    
    while time.time() - start_time < time_limit:
        # Pick random index to flip
        i = rng.randint(0, N - 1)
        
        # Flip
        old_val = best_s[i]
        new_val = -old_val
        best_s[i] = new_val
        
        # Calculate new score
        # Optimization: Only sums of APs containing i change.
        # But with sparse dot, we just recalculate all. 
        # A.dot is fast.
        new_sums = A.dot(best_s)
        new_score = np.max(np.abs(new_sums))
        
        delta = new_score - best_score
        
        if delta < 0 or (T > 0 and rng.random() < np.exp(-delta / T)):
            best_score = new_score
        else:
            # Revert
            best_s[i] = old_val
            
        T *= cooling
        
    return [int(x) for x in best_s]
