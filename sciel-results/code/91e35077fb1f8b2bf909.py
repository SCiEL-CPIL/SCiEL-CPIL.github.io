import numpy as np

def construct():
    start_time = time.time()
    max_val = 1000001
    lpf = np.zeros(max_val + 1, dtype=np.int32)
    
    # Sieve
    # We need to iterate primes.
    # Efficient way:
    # 1. Create array of indices.
    # 2. For each prime p, update multiples.
    
    # Standard numpy sieve for LPF is tricky because we need to update in place.
    # We can loop over primes in Python, but use numpy slicing for updates.
    
    # Identify primes
    # A simple way is to use the fact that lpf[i]==0 means prime (initially).
    # But numpy doesn't support dynamic "if lpf[p]==0" inside a loop easily without masking.
    
    # Alternative:
    # 1. Initialize lpf with 0.
    # 2. For p from 2 to max_val:
    #    if lpf[p] == 0:
    #       lpf[p::p] = p
    # This is O(N log log N) but with fast numpy slicing.
    
    for p in range(2, max_val + 1):
        if lpf[p] == 0:
            lpf[p::p] = p
            
    # Scan
    # Use numpy to compute max(lpf[n], lpf[n+1]) for all n.
    # scores = np.maximum(lpf[10000:1000001], lpf[100001:1000002])
    # min_score = np.min(scores)
    # best_n = 10000 + np.argmin(scores)
    
    scores = np.maximum(lpf[10000:1000001], lpf[100001:1000002])
    min_score = np.min(scores)
    best_n = 10000 + np.argmin(scores)
    
    return int(best_n)
