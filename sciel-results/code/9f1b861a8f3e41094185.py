import numpy as np

def construct():
    MIN_N = 10000
    MAX_N = 1000000
    
    # Sieve for primes up to sqrt(MAX_N) is sufficient for the loop structure
    # But we need to know which numbers are divisible by p^e.
    # We can just iterate p from 2 to isqrt(MAX_N) and check if prime.
    
    # Simple prime check or sieve
    limit = int(np.sqrt(MAX_N)) + 1
    is_prime = np.ones(limit + 1, dtype=bool)
    is_prime[0:2] = False
    for i in range(2, int(np.sqrt(limit)) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = False
    
    primes = np.where(is_prime)[0]
    
    pp = np.ones(MAX_N + 1, dtype=np.int64)
    indices = np.arange(1, MAX_N + 1, dtype=np.int64)
    
    for p in primes:
        p_int = int(p)
        # Max e such that p^e <= MAX_N
        # e can be 2, 3, ...
        # We process e from high to low? Or low to high?
        # The condition is v_p(n) == e.
        # These sets are disjoint for fixed p.
        # So order doesn't matter for correctness, but we must cover all e >= 2.
        
        # Find max e
        # p^2 <= MAX_N
        if p_int * p_int > MAX_N:
            continue
            
        # Calculate powers
        p_e = p_int * p_int
        e = 2
        
        while p_e <= MAX_N:
            # Next power
            p_e_plus_1 = p_e * p_int
            
            # Mask for v_p(n) == e
            # n % p_e == 0 AND n % p_e_plus_1 != 0
            # Note: if p_e_plus_1 > MAX_N, then n % p_e_plus_1 is just n (if n < p_e_plus_1) or n % p_e_plus_1.
            # If p_e_plus_1 > MAX_N, then no n in [1, MAX_N] is divisible by p_e_plus_1.
            # So mask is simply (indices % p_e == 0)
            
            mask = (indices % p_e == 0)
            if p_e_plus_1 <= MAX_N:
                mask = mask & (indices % p_e_plus_1 != 0)
            
            pp[indices[mask]] *= p_e
            
            e += 1
            p_e = p_e_plus_1
            
    # Find max product
    # pp[n] * pp[n+1] for n in [MIN_N, MAX_N-1]
    # Slice pp
    pp_slice = pp[MIN_N:MAX_N]
    pp_slice_next = pp[MIN_N+1:MAX_N+1]
    
    products = pp_slice * pp_slice_next
    best_idx = np.argmax(products)
    best_n = MIN_N + best_idx
    
    return int(best_n)
