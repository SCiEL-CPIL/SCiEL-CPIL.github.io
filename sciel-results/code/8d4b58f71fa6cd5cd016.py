import time
import numpy as np
from collections import Counter

# EVOLVE-BLOCK-START
def construct():
    start = time.time()
    DEADLINE = 85  # safety margin before 90s kill
    
    N = 1000000 + 24  # max index needed: n+24 with n up to 1e6
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23]
    
    # Precompute 23-smooth part for 0..N using NumPy
    sf = np.arange(N + 1, dtype=np.int64)
    for p in primes:
        pk = p
        while pk <= N:
            sf[pk::pk] //= p
            pk *= p
    
    if time.time() - start > DEADLINE:
        return 1000
    
    # Structural change: Use a two-phase search.
    # Phase 1: Vectorized coarse scan in chunks to identify promising regions.
    # Phase 2: Fine-grained sliding window on top candidates.
    
    sf_list = sf.tolist()
    
    # Phase 1: Sample every K-th window to find low-distinct regions
    K = 100  # step size for coarse scan
    best_count = float('inf')
    best_n = 1000
    candidates = []
    
    # Coarse scan: check windows at positions 1000, 1000+K, 1000+2K, ...
    for n in range(1000, 1000001, K):
        # Quick distinct count using set
        window = sf_list[n+1:n+25]
        d = len(set(window))
        if d < best_count:
            best_count = d
            best_n = n
        # Collect candidates near low-distinct windows
        if d <= best_count + 2:
            candidates.append(n)
    
    if time.time() - start > DEADLINE:
        return best_n
    
    # Phase 2: Fine scan around each candidate region
    fine_best_count = best_count
    fine_best_n = best_n
    
    for cand in candidates:
        lo = max(1000, cand - K)
        hi = min(1000000, cand + K)
        
        if time.time() - start > DEADLINE:
            break
        
        # Sliding window fine scan in this region
        if lo > 1000:
            # Jump to lo by computing distinct directly
            window = sf_list[lo+1:lo+25]
            d = len(set(window))
            if d < fine_best_count:
                fine_best_count = d
                fine_best_n = lo
        else:
            window = sf_list[1001:1025]
            d = len(set(window))
            if d < fine_best_count:
                fine_best_count = d
                fine_best_n = 1000
        
        win = {}
        for v in sf_list[lo+1:lo+25]:
            win[v] = win.get(v, 0) + 1
        
        for n in range(lo + 1, hi + 1):
            old_idx = n
            new_idx = n + 24
            
            c_old = win[sf_list[old_idx]]
            c_old -= 1
            if c_old == 0:
                del win[sf_list[old_idx]]
            else:
                win[sf_list[old_idx]] = c_old
            
            c_new = win.get(sf_list[new_idx], 0)
            win[sf_list[new_idx]] = c_new + 1
            
            d = len(win)
            if d < fine_best_count:
                fine_best_count = d
                fine_best_n = n
                if fine_best_count == 1:
                    return fine_best_n
            
            if time.time() - start > DEADLINE:
                return fine_best_n
    
    return fine_best_n
# EVOLVE-BLOCK-END
