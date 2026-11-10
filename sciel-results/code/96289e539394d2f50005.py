import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 75  # Leave 15s margin for evaluator overhead
    
    # Upper bound for values we need LPF for is 1000000 + 1 = 1000001
    max_val = 1000001
    
    # Use numpy array for fast vectorized sieve operations
    # int32 is sufficient since max_val < 2^31
    lpf = np.zeros(max_val + 1, dtype=np.int32)
    
    # Sieve to find largest prime factors
    # For each prime p, assign p to all multiples of p.
    # Since we iterate p in increasing order, the last assignment (largest p) remains.
    for p in range(2, max_val + 1):
        if lpf[p] == 0:
            # p is prime
            # Vectorized assignment: lpf[p::p] = p
            lpf[p::p] = p
            
        # Time check every 100000 iterations to avoid overhead
        if p % 100000 == 0 and time.time() > deadline:
            return 10000
    
    # Final time check
    if time.time() > deadline:
        return 10000
    
    # Vectorized computation of scores for all n in [10000, 1000000]
    # score(n) = max(lpf[n], lpf[n+1])
    # n ranges from 10000 to 1000000
    # lpf[n] corresponds to lpf[10000:1000001]
    # lpf[n+1] corresponds to lpf[10001:1000002]
    
    lpf_n = lpf[10000:1000001]
    lpf_np1 = lpf[10001:1000002]
    
    scores = np.maximum(lpf_n, lpf_np1)
    
    if len(scores) == 0:
        return 10000
        
    # Find the index of the minimum score
    min_score_idx = np.argmin(scores)
    min_score = scores[min_score_idx]
    best_n = 10000 + min_score_idx
    
    return int(best_n)
