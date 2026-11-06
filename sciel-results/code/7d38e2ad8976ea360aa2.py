import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 50

    N_MIN = 10000
    N_MAX = 1000000

    # Strategy: Find n in [N_MIN, N_MAX] minimizing max(lpf(n), lpf(n+1))
    # lpf(x) = largest prime factor of x
    
    # Step 1: Sieve primes up to sqrt(N_MAX+1) ≈ 1001
    # Then compute lpf for all numbers in [N_MIN, N_MAX+1]
    
    limit = N_MAX + 1  # we need lpf up to N_MAX+1
    
    # Sieve of Eratosthenes for smallest prime factor
    spf = np.zeros(limit + 1, dtype=np.int32)
    for i in range(2, limit + 1):
        if spf[i] == 0:
            spf[i::i] = i
    
    # Compute largest prime factor using spf
    # For each number, repeatedly divide by spf to get the largest prime factor
    # Alternative: for each prime p, for each multiple of p, lpf[multiple] = max(lpf[multiple], p)
    # This is O(n log log n) with numpy
    
    lpf = np.zeros(limit + 1, dtype=np.int32)
    # For each prime, mark its multiples
    for p in range(2, limit + 1):
        if spf[p] == p:  # p is prime
            for multiple in range(p, limit + 1, p):
                if p > lpf[multiple]:
                    lpf[multiple] = p
    
    # Check time
    if time.time() > deadline - 5:
        # Fallback: just return a reasonable value
        return 10080  # 10080 = 2^5 * 3^2 * 5 * 7, lpf=7
    
    # Compute scores for n in [N_MIN, N_MAX]
    n_range = np.arange(N_MIN, N_MAX + 1)
    scores = np.maximum(lpf[n_range], lpf[n_range + 1])
    
    # Find the minimum
    best_idx = np.argmin(scores)
    best_n = int(n_range[best_idx])
    best_score = int(scores[best_idx])
    
    return best_n
