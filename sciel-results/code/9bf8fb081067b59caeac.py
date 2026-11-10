import time
import math
import numpy as np

def construct():
    start_time = time.time()
    DEADLINE = 85.0  # Safety margin before the 90s kill

    N_MIN = 10000
    N_MAX = 1000000
    M = N_MAX + 1  # We need lpf for values up to 1000001

    # 1. Sieve of Eratosthenes using numpy to find primes
    is_prime = np.ones(M + 1, dtype=bool)
    is_prime[0] = False
    is_prime[1] = False
    limit = math.isqrt(M) + 1
    for i in range(2, limit):
        if is_prime[i]:
            is_prime[i*i::i] = False

    # Collect primes
    primes = np.where(is_prime)[0]
    
    # 2. Compute largest prime factor for each number up to M
    # lpf[x] = largest prime factor of x
    # We use int32 to save memory and improve cache performance.
    # The maximum prime factor for a number <= 1,000,001 is 1,000,001 (if prime), 
    # which fits in int32.
    lpf = np.zeros(M + 1, dtype=np.int32)
    
    # Iterate through primes in increasing order.
    # For each prime p, set lpf[m] = p for all multiples m of p.
    # Since we iterate p in increasing order, the last assignment for any m
    # will be by the largest prime factor of m.
    for p in primes:
        p_int = int(p)
        # Generate multiples of p
        # np.arange is efficient for this
        multiples = np.arange(p_int, M + 1, p_int, dtype=np.int32)
        lpf[multiples] = p_int
        
    # Time check: if we are somehow running out of time (unlikely for this size), 
    # we might skip the final search if it's not started, but the search is vectorized 
    # and very fast.
    if time.time() - start_time > DEADLINE - 5.0:
        # Fallback: return a known good candidate or the start of the range
        # The exhaustive search below is ~1ms, so this is just a safety net.
        return N_MIN

    # 3. Find n in [N_MIN, N_MAX] minimizing max(lpf[n], lpf[n+1])
    # Vectorized search
    # lpf[n] for n in [N_MIN, N_MAX]
    lpf_n = lpf[N_MIN : N_MAX + 1]
    # lpf[n+1] for n in [N_MIN, N_MAX] -> indices [N_MIN+1, N_MAX+1]
    lpf_np1 = lpf[N_MIN + 1 : N_MAX + 2]
    
    # Compute max for each pair
    max_lpf = np.maximum(lpf_n, lpf_np1)
    
    # Find the index of the minimum value
    min_idx = np.argmin(max_lpf)
    best_n = N_MIN + min_idx
    best_val = max_lpf[min_idx]
    
    # Optional: Verify the result with a simple loop to ensure no numpy edge case issues
    # This is a sanity check. In practice, the vectorized result is correct.
    # We only do this if we have plenty of time left.
    if time.time() - start_time < DEADLINE - 10.0:
        # Re-verify best_n
        val_check = max(int(lpf[best_n]), int(lpf[best_n + 1]))
        if val_check != int(best_val):
            # If mismatch, fall back to a linear scan (should never happen)
            best_val = float('inf')
            for n in range(N_MIN, N_MAX + 1):
                v = max(int(lpf[n]), int(lpf[n+1]))
                if v < best_val:
                    best_val = v
                    best_n = n

    return int(best_n)
