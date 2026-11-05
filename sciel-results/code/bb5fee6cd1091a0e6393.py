import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 80  # safety margin within 90s limit

    # We need largest prime factor for values up to 1,000,001 (n+1 when n=1,000,000)
    MAXV = 1000001

    # Step 1: Generate primes up to MAXV using Sieve of Eratosthenes
    # This is faster than iterating all integers and checking lpf[p]==0
    is_prime = np.ones(MAXV + 1, dtype=bool)
    is_prime[0] = False
    is_prime[1] = False
    
    # Standard sieve
    for i in range(2, int(MAXV**0.5) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = False
            
    # Get the list of primes
    primes = np.where(is_prime)[0]
    
    # Step 2: Build largest-prime-factor array
    # lpf[i] will hold the largest prime factor of i.
    lpf = np.zeros(MAXV + 1, dtype=np.int64)
    
    # Iterate only over primes
    # Since we iterate p in increasing order, the last assignment wins,
    # giving the largest prime factor.
    for p in primes:
        # Mark all multiples of p
        lpf[p::p] = p

    # Step 3: Find n in [10000, 1000000] minimizing max(lpf[n], lpf[n+1])
    # Use vectorized numpy for speed
    n_arr = np.arange(10000, 1000001, dtype=np.int64)
    lpf_n = lpf[n_arr]
    lpf_np1 = lpf[n_arr + 1]
    max_lpf = np.maximum(lpf_n, lpf_np1)

    # Find the index of the minimum
    min_idx = np.argmin(max_lpf)
    best_n = int(n_arr[min_idx])
    
    # Verify we're within time budget
    if time.time() > deadline:
        pass  # still return best found

    return best_n
