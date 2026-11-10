"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import time

def construct():
    N = 256
    start_time = time.time()
    
    # Find all primes up to N
    is_prime = [True] * (N + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(N**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, N+1, i):
                is_prime[j] = False
    primes = [p for p in range(2, N+1) if is_prime[p]]
    prime_idx = {p: i for i, p in enumerate(primes)}
    
    # For each number 2..N, find its prime factors
    prime_factors = {}
    for n in range(2, N+1):
        factors = set()
        x = n
        for p in primes:
            if p * p > x:
                break
            if x % p == 0:
                factors.add(p)
                while x % p == 0:
                    x //= p
        if x > 1:
            factors.add(x)
        prime_factors[n] = factors
    
    # Precompute bitmask for each number
    num_mask = {}
    for n in range(2, N+1):
        m = 0
        for pf in prime_factors[n]:
            m |= (1 << prime_idx[pf])
        num_mask[n] = m
    
    # Process primes in descending order
    primes_desc = sorted(primes, reverse=True)
    
    # For each prime, get top candidates by value
    prime_cands = {}
    for p in primes_desc:
        cands = [n for n in range(2, N+1) if p in prime_factors[n]]
        cands.sort(reverse=True)
        # Limit to top 5 for search efficiency
        prime_cands[p] = cands[:5]
    
    # Max prime power for each prime
    max_pp = {}
    for p in primes:
        v = p
        while v * p <= N:
            v *= p
        max_pp[p] = v
    
    # Suffix sum for upper bound
    suffix = [0] * (len(primes_desc) + 1)
    for i in range(len(primes_desc) - 1, -1, -1):
        suffix[i] = suffix[i+1] + max_pp[primes_desc[i]]
    
    best_sum = 1
    best_sel = [1]
    
    def bnb(idx, used_mask, curr_sum, curr_sel):
        nonlocal best_sum, best_sel
        
        if curr_sum > best_sum:
            best_sum = curr_sum
            best_sel = list(curr_sel)
        
        if idx == len(primes_desc):
            return
        
        # Upper bound
        if curr_sum + suffix[idx] <= best_sum:
            return
        
        # Time check
        if time.time() - start_time > 55:
            return
        
        p = primes_desc[idx]
        
        # Option: skip
        bnb(idx + 1, used_mask, curr_sum, curr_sel)
        
        # Option: try candidates
        for n in prime_cands[p]:
            if num_mask[n] & used_mask:
                continue
            bnb(idx + 1, used_mask | num_mask[n], curr_sum + n, curr_sel + [n])
    
    bnb(0, 0, 1, [1])
    return best_sel

# EVOLVE-BLOCK-END
