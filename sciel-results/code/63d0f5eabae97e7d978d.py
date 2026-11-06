import time
import math
import numpy as np

def construct():
    start_time = time.time()
    deadline = 85  # safety margin before 90s kill
    
    N_MIN = 10000
    N_MAX = 1000000
    
    # Precompute largest prime factor for all values up to N_MAX + 1
    max_val = N_MAX + 1
    lpf = np.zeros(max_val + 1, dtype=np.int64)
    
    # Sieve: for each prime p, set lpf[p] = p, and for multiples, update lpf
    # Use a sieve approach where we iterate through primes and mark their multiples
    lpf[:] = 1  # initialize
    
    for i in range(2, max_val + 1):
        if lpf[i] == 1:  # i is prime
            lpf[i] = i
            for j in range(i * i, max_val + 1, i):
                lpf[j] = i  # i is the smallest prime factor, but we want largest
                # Actually, we need to track the largest prime factor
                # Let's use a different approach: for each number, its LPF is the largest prime that divides it
                # We can do this by iterating through primes and for each multiple, setting lpf[multiple] = max(lpf[multiple], prime)
    
    # Better approach: for each prime p, for each multiple m of p, set lpf[m] = max(lpf[m], p)
    # But this is O(N log N) which might be slow. Let's use the SPF approach and compute LPF from it.
    
    # Actually, let's just compute LPF directly using a sieve-like method:
    # lpf[x] = largest prime factor of x
    # For prime p, lpf[p] = p
    # For composite x, lpf[x] = max(lpf[x // p] for p | x)
    # But this is recursive. Instead, let's use:
    # lpf[x] = max of all primes p such that p divides x
    
    # Efficient method: initialize lpf to 1, then for each prime p, for each multiple m, lpf[m] = max(lpf[m], p)
    # This works because the largest prime factor of m is the largest prime that divides m.
    
    lpf = np.ones(max_val + 1, dtype=np.int64)
    for i in range(2, max_val + 1):
        if lpf[i] == 1:  # i is prime
            lpf[i] = i
            for j in range(i, max_val + 1, i):
                lpf[j] = i  # Since we iterate i in increasing order, the last prime that divides j is the largest
    
    # Wait, that's wrong. If we iterate i from 2 to max_val, and for each prime i, we set lpf[j] = i for all multiples j,
    # then lpf[j] will be set to the LARGEST prime factor because larger primes are processed later.
    # But we only process i if lpf[i] == 1, which means i is prime. So this should work!
    # However, this is O(N log N) which might be too slow for N = 10^6.
    
    # Let's check: sum over primes p of N/p ≈ N * log(log(N)) ≈ 10^6 * log(log(10^6)) ≈ 10^6 * 2.6 ≈ 2.6 * 10^6
    # That's about 2.6 million operations, which should be fine.
    
    # But the inner loop in Python is slow. Let's use numpy for the inner loop.
    lpf = np.ones(max_val + 1, dtype=np.int64)
    for i in range(2, max_val + 1):
        if lpf[i] == 1:  # i is prime
            lpf[i] = i
            # Set lpf[j] = i for all j = i, 2i, 3i, ...
            # Using numpy slice assignment
            lpf[i::i] = i
    
    # Now compute the objective for each n in [N_MIN, N_MAX]
    # val[n] = max(lpf[n], lpf[n+1])
    n_arr = np.arange(N_MIN, N_MAX + 1)
    lpf_n = lpf[n_arr]
    lpf_np1 = lpf[n_arr + 1]
    vals = np.maximum(lpf_n, lpf_np1)
    
    best_idx = np.argmin(vals)
    best_n = int(n_arr[best_idx])
    best_val = int(vals[best_idx])
    
    return best_n
