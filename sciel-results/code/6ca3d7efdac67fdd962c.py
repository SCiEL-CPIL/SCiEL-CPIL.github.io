import math
import time
import random

def construct():
    start_time = time.time()
    LIMIT = 115.0 # Safety margin for 150s kill
    MAX_S = 10**12
    MAX_K_SEARCH = 300 # Search window size to find max run
    
    # 1. Generate primes up to 10^6
    limit_p = 10**6
    is_prime = [True] * (limit_p + 1)
    is_prime[0] = False
    is_prime[1] = False
    for i in range(2, int(limit_p**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, limit_p + 1, i):
                is_prime[j] = False
    primes = [i for i, val in enumerate(is_prime) if val]
    
    # 2. Batch Phi Calculation
    # Computes phi for n, n+1, ..., n+length-1
    def compute_phis(n, length):
        # vals: the numbers themselves
        # phis: current phi values
        # rems: remaining part after removing small prime factors
        vals = [n + i for i in range(length)]
        phis = vals[:]
        rems = vals[:]
        
        # Sieve for small primes
        for p in primes:
            if p * p > n + length:
                # Optimization: if p^2 > max_val, we can break? 
                # No, we need to process all primes up to sqrt(max_val) to ensure 
                # that any remaining factor > 1 is prime.
                # Actually, if p > sqrt(max_val), then any multiple of p in the range 
                # is p * k where k < sqrt(max_val). 
                # But our primes list goes up to 10^6. sqrt(10^12) = 10^6.
                # So we iterate all primes in the list.
                pass
            
            # Find the first multiple of p in [n, n+length-1]
            # start = ceil(n/p) * p
            start = ((n + p - 1) // p) * p
            
            # Iterate through multiples
            for m in range(start, n + length, p):
                idx = m - n
                if phis[idx] != 0: # Optimization: if already 0? No, phi is never 0 for n>=1
                    # Subtract the factor
                    phis[idx] -= phis[idx] // p
                    # Remove p from remaining
                    while rems[idx] % p == 0:
                        rems[idx] //= p
        
        # Handle large prime factors
        for i in range(length):
            if rems[i] > 1:
                # rems[i] is a prime factor
                phis[i] -= phis[i] // rems[i]
                
        return phis

    best_k = 1
    best_s = 1
    
    rng = random.Random(42)
    
    # Search loop
    while time.time() - start_time < LIMIT:
        # Random start
        max_start = MAX_S - MAX_K_SEARCH + 1
        if max_start < 1:
            break
        s = rng.randint(1, max_start)
        
        phis = compute_phis(s, MAX_K_SEARCH)
        
        # Find longest run of distinct values
        seen = {}
        l = 0
        max_len = 0
        max_start_idx = 0
        for r in range(MAX_K_SEARCH):
            if phis[r] in seen and seen[phis[r]] >= l:
                l = seen[phis[r]] + 1
            else:
                if r - l + 1 > max_len:
                    max_len = r - l + 1
                    max_start_idx = l
            seen[phis[r]] = r
            
        if max_len > best_k:
            best_k = max_len
            best_s = s + max_start_idx
            
        # If we found a very long run, we can be more optimistic or just continue
        # The max possible is 200. If we find 200, we can return early?
        # The task says "Maximize k". If we find 200, it's the max allowed.
        if best_k >= 200:
            break

    return {'start': best_s, 'length': best_k}
