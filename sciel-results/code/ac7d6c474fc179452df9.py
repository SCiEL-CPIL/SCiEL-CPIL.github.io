import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 80  # Leave 10s margin
    
    max_val = 1000001
    # Use int32 to save memory and potentially improve cache performance
    lpf = np.zeros(max_val + 1, dtype=np.int32)
    
    # Sieve for Largest Prime Factor
    # Iterate p from 2 to max_val
    # If lpf[p] == 0, p is prime
    # Assign p to all multiples of p
    for p in range(2, max_val + 1):
        if lpf[p] == 0:
            lpf[p::p] = p
            
        # Time check every 10000 primes/iterations? 
        # Actually, the loop is fast. We can check less frequently or rely on the fact that it's fast.
        # But let's keep a safety check.
        if p % 100000 == 0:
            if time.time() > deadline:
                break

    if time.time() > deadline:
        return 10000

    # We want to minimize max(lpf[n], lpf[n+1]) for n in [10000, 1000000]
    # This corresponds to indices 10000 to 1000000 in lpf
    # n+1 corresponds to indices 10001 to 1000001
    
    # Vectorized calculation of scores
    # lpf[n] is lpf[10000:1000001]
    # lpf[n+1] is lpf[10001:1000002]
    
    scores = np.maximum(lpf[10000:1000001], lpf[10001:1000002])
    
    if len(scores) == 0:
        return 10000
        
    min_score_idx = np.argmin(scores)
    min_score = scores[min_score_idx]
    best_n = 10000 + min_score_idx
    
    return int(best_n)
