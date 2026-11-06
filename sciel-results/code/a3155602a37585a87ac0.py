"""Improved program using a modified sieve to compute powerful parts efficiently."""
import time
import math
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    """
    Return an integer n in [10000, 1000000] that maximizes
    powerful_part(n) * powerful_part(n+1).
    
    The powerful part of an integer is the product of all prime powers p^e
    in its factorization where e >= 2.
    """
    start_time = time.time()
    deadline = start_time + 75.0  # Safety margin before 90s limit
    
    N = 1000000
    LOWER = 10000
    
    # powerful[i] will hold the powerful part of i
    # Initialize to 1 (empty product)
    powerful = np.ones(N + 2, dtype=np.int64)
    
    # Sieve: for each prime p, for each power p^e with e >= 2,
    # multiply powerful[k] by p^e for all k that are multiples of p^e
    # but NOT multiples of p^(e+1) (to get the FULL prime power in factorization)
    
    # Actually, the powerful part is the product of p^e for each prime p where
    # the exponent of p in the factorization of n is e >= 2.
    # So for each prime p and each e >= 2, for each multiple of p^e but not p^(e+1),
    # multiply powerful[n] by p^e.
    
    # Better approach: for each prime p, compute the highest power p^e that divides n
    # for each n, and if e >= 2, multiply powerful[n] by p^e.
    
    # Even better: for each prime p, let p^2, p^3, ..., p^k be the powers.
    # For each n divisible by p^j (j >= 2), the contribution is p^j where j is
    # the exact exponent. We can do this by:
    # 1. For each p, start with p^2. For multiples of p^2, mark them.
    # 2. For multiples of p^3, the contribution should be p^3, not p^2 * p^3.
    # 
    # Cleanest approach: for each prime p, for each n, determine v_p(n) (the exponent
    # of p in n). If v_p(n) >= 2, multiply powerful[n] by p^(v_p(n)).
    #
    # Vectorized: for each prime p, iterate through powers p^2, p^3, ...
    # For each power p^e, for indices that are multiples of p^e but NOT multiples of p^(e+1),
    # multiply by p^e.
    
    # Generate primes up to sqrt(N) (since p^2 <= N implies p <= sqrt(N))
    # Actually we need primes up to N, but only p where p^2 <= N matter for e >= 2
    # p^2 <= N => p <= sqrt(N) ~ 1000
    
    limit = int(math.isqrt(N)) + 1
    
    # Simple sieve for primes up to limit
    is_prime = np.ones(limit + 1, dtype=bool)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(math.isqrt(limit)) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = False
    
    primes = np.where(is_prime)[0]
    
    for p in primes:
        p2 = p * p
        if p2 > N:
            break
        
        # For this prime p, we need to find for each n the exact exponent v_p(n)
        # and if v_p(n) >= 2, multiply powerful[n] by p^(v_p(n))
        
        # Strategy: 
        # multiples of p^2: indices p^2, 2*p^2, 3*p^2, ...
        # multiples of p^3: indices p^3, 2*p^3, 3*p^3, ...
        # etc.
        
        # For n that is a multiple of p^e but not p^(e+1), contribution is p^e.
        
        # We'll work with arrays of indices.
        # Start with all multiples of p^2
        power = p2
        next_power = power * p
        
        # Indices that are multiples of power but not next_power
        # These get contribution p^2 (i.e., power)
        
        while power <= N:
            if time.time() > deadline:
                break
            
            # Multiples of 'power' up to N
            # indices: power, 2*power, 3*power, ..., floor(N/power)*power
            count = N // power
            if count == 0:
                break
            
            # Indices that are multiples of power
            indices = np.arange(power, N + 1, power, dtype=np.int64)
            
            if next_power <= N:
                # Remove those that are also multiples of next_power
                # Multiples of next_power: next_power, 2*next_power, ...
                # An index i is a multiple of both power and next_power iff it's a multiple of next_power
                # (since next_power is a multiple of power)
                count_next = N // next_power
                if count_next > 0:
                    next_indices = np.arange(next_power, N + 1, next_power, dtype=np.int64)
                    # Create a boolean mask for indices
                    mask = np.ones(len(indices), dtype=bool)
                    # next_indices are a subset of indices (every p-th element)
                    # indices[j] = (j+1)*power
                    # next_indices[k] = (k+1)*next_power = (k+1)*p*power = ((k+1)*p)*power
                    # So next_indices[k] corresponds to indices[(k+1)*p - 1]
                    # i.e., indices[p-1], indices[2p-1], indices[3p-1], ...
                    # Mark these as False
                    mask[p-1::p] = False
                    indices = indices[mask]
            
            if len(indices) > 0:
                powerful[indices] *= power
            
            power = next_power
            next_power = power * p
            if power > N:
                break
    
    if time.time() > deadline:
        # Fall back to a simple answer
        return LOWER
    
    # Now find n in [LOWER, N] that maximizes powerful[n] * powerful[n+1]
    # Note: n can be up to N, but n+1 can be up to N+1
    # We need powerful[n+1] for n up to N, so we need powerful up to N+1
    # But our sieve only computed up to N. Let's handle n+1 = N+1 separately.
    # Actually, let's just consider n in [LOWER, N-1] so n+1 <= N.
    # Or extend to N. Let's do n in [LOWER, N-1] to be safe.
    
    # products[i] = powerful[i] * powerful[i+1] for i in [LOWER, N-1]
    products = powerful[LOWER:N] * powerful[LOWER+1:N+1]
    
    best_idx = np.argmax(products)
    best_n = LOWER + best_idx
    best_val = products[best_idx]
    
    return int(best_n)
# EVOLVE-BLOCK-END
