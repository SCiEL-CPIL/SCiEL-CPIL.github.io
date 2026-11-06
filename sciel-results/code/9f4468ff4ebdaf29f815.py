import numpy as np
import time

def construct():
    start_time = time.time()
    MAX_N = 1000000
    LIMIT = MAX_N + 1
    
    # 1. Sieve of Eratosthenes to find primes up to LIMIT
    is_prime = np.ones(LIMIT + 1, dtype=bool)
    is_prime[0] = False
    is_prime[1] = False
    
    for i in range(2, int(np.sqrt(LIMIT)) + 1):
        if is_prime[i]:
            is_prime[i*i : LIMIT+1 : i] = False
            
    primes = np.where(is_prime)[0]
    
    # 2. Compute Largest Prime Factor (LPF)
    lpf = np.arange(LIMIT + 1, dtype=np.int64)
    
    # Iterate through primes in increasing order.
    # For each prime p, update lpf for all multiples of p.
    # Since we go in increasing order, the last prime to update a number m
    # will be its largest prime factor.
    for p in primes:
        # Optimization: if p*p > LIMIT, we only need to update lpf[p] itself?
        # No, p is prime, so lpf[p] = p.
        # But p might be a factor of a larger number m < LIMIT.
        # However, if p > sqrt(LIMIT), then p*p > LIMIT.
        # The only multiple of p in range is p itself.
        # So we can break early?
        # If p > sqrt(LIMIT), the slice p::p only contains p.
        # So we can just set lpf[p] = p (which it already is) and skip the rest?
        # Actually, if p > sqrt(LIMIT), it has no multiples in [2p, LIMIT].
        # So the slice is just [p].
        # We can optimize by only looping p <= sqrt(LIMIT) for the "fill" part,
        # but primes > sqrt(LIMIT) are already correctly set to themselves in lpf.
        # Wait, if m is composite, it has a factor <= sqrt(m).
        # So the largest prime factor of m is either m (if prime) or
        # determined by factors.
        # If m is composite, its largest prime factor P_LPF satisfies P_LPF <= m/2.
        # But we need to set lpf[m] correctly.
        # The current loop handles all primes.
        # To speed up, we can stop updating multiples for p > sqrt(LIMIT)
        # because there are no multiples other than p itself.
        # And lpf[p] is already p.
        # So we can break the loop when p * p > LIMIT.
        
        if p * p > LIMIT:
            break
            
        lpf[p : LIMIT+1 : p] = p
        
    # Wait, if I break early, do I miss setting lpf for numbers whose LARGEST
    # prime factor is > sqrt(LIMIT)?
    # Example: m = 2 * 500003 (prime).
    # sqrt(LIMIT) ~ 1000.
    # 500003 > 1000.
    # The loop breaks at p > 1000.
    # So lpf[1000006] is never updated by 500003.
    # It was updated by 2, so lpf[1000006] = 2.
    # But the LARGEST prime factor is 500003.
    # So this optimization is WRONG.
    
    # We MUST iterate through all primes to correctly set the LARGEST factor.
    # Because a number can be 2 * LargePrime.
    
    # So we revert to iterating all primes.
    # The cost is acceptable.
    
    # Re-do LPF calculation correctly
    lpf = np.arange(LIMIT + 1, dtype=np.int64)
    for p in primes:
        lpf[p : LIMIT+1 : p] = p
        
    # 3. Find n in [10000, MAX_N] minimizing max(lpf[n], lpf[n+1])
    n_start = 10000
    n_end = MAX_N
    
    vals_n = lpf[n_start : n_end + 1]
    vals_np1 = lpf[n_start + 1 : n_end + 2]
    
    max_lpf = np.maximum(vals_n, vals_np1)
    
    min_val = np.min(max_lpf)
    best_idx = np.argmin(max_lpf)
    best_n = n_start + best_idx
    
    return int(best_n)
