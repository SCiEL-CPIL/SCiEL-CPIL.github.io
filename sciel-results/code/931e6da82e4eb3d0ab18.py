"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 80  # Safety margin before 90s kill
    
    lo = 10000
    hi = 1000000
    limit = hi + 1  # We need powerful part for hi+1 as well
    
    # 1. Sieve for primes up to sqrt(limit)
    # Primes p such that p^2 <= limit
    max_p = int(np.sqrt(limit)) + 1
    is_prime = np.ones(max_p + 1, dtype=bool)
    is_prime[0] = False
    is_prime[1] = False
    for i in range(2, int(np.sqrt(max_p)) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = False
    
    primes = np.where(is_prime)[0]
    
    # 2. Initialize powerful part array
    # pp[n] will hold the powerful part of n
    # Using int64 to be safe, though values are <= 10^6
    pp = np.ones(limit + 1, dtype=np.int64)
    
    # 3. Update pp using vectorized operations
    for p in primes:
        p2 = p * p
        if p2 > limit:
            break
            
        # For each power p^k (k >= 2), we need to multiply pp[m] by p 
        # for every m that is a multiple of p^k but NOT p^(k+1)?
        # Actually, the contribution is p^e where e is the exponent.
        # If we multiply by p for every multiple of p^2, p^3, p^4...
        # Let's trace:
        # m is multiple of p^2. We want to add p^2 to the product? No.
        # The powerful part includes p^e.
        # If we do:
        # for k in range(2, max_k+1):
        #    pp[multiples of p^k] *= p
        # Then for a number m with v_p(m) = e:
        # It is a multiple of p^2, p^3, ..., p^e.
        # It gets multiplied by p (e-1) times.
        # Initial value 1. Result p^(e-1).
        # This is incorrect. We want p^e.
        
        # Correction:
        # We want to multiply by p^e.
        # We can do this by:
        # 1. For all multiples of p^2, multiply by p^2.
        # 2. For all multiples of p^3, multiply by p. (Total p^3)
        # 3. For all multiples of p^4, multiply by p. (Total p^4)
        # ...
        # This works.
        
        # Step 1: Multiply by p^2 for all multiples of p^2
        multiples_p2 = np.arange(p2, limit + 1, p2)
        pp[multiples_p2] *= p2
        
        # Step 2+: Multiply by p for all multiples of p^3, p^4, ...
        p_k = p2 * p # p^3
        while p_k <= limit:
            multiples_pk = np.arange(p_k, limit + 1, p_k)
            pp[multiples_pk] *= p
            p_k *= p
            
        if time.time() > deadline:
            break
    
    # 4. Calculate products pp[n] * pp[n+1] for n in [lo, hi-1]
    # n goes from lo to hi-1 (since n+1 <= hi)
    # pp indices: lo to hi
    # We need pp[lo] * pp[lo+1], ..., pp[hi-1] * pp[hi]
    
    # Slice the array
    # left part: pp[lo : hi]  (indices lo, lo+1, ..., hi-1)
    # right part: pp[lo+1 : hi+1] (indices lo+1, ..., hi)
    
    left = pp[lo:hi]
    right = pp[lo+1:hi+1]
    
    products = left * right
    
    if time.time() > deadline:
        # Fallback to a simple value if we ran out of time during computation
        # But the vectorized loop is very fast, this is unlikely.
        return lo
    
    max_idx = np.argmax(products)
    best_n = lo + max_idx
    
    return best_n
# EVOLVE-BLOCK-END
