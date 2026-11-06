import numpy as np
import time

def construct():
    start_time = time.time()
    N = 1000000
    # We need pp for n and n+1, so up to 1000000.
    # n is in [10000, 1000000]. If n=1000000, we need pp[1000000] and pp[1000001]?
    # The task says "Return an integer n in [10000, 1000000]".
    # "Maximize powerful_part(n)*powerful_part(n+1)".
    # If n=1000000, we need powerful_part(1000001).
    # So we should compute pp up to 1000001.
    
    MAX_VAL = 1000001
    pp = np.ones(MAX_VAL + 1, dtype=np.int64)
    
    # Sieve for primes up to sqrt(MAX_VAL) or just iterate up to MAX_VAL
    # We only need primes p where p^2 <= MAX_VAL, i.e., p <= 1000.
    # Primes > 1000 have p^2 > 1000000, so they don't contribute to powerful part
    # (exponent would be 1, which is not >= 2).
    
    # Simple sieve for primes up to 1000
    limit = int(MAX_VAL ** 0.5) + 1
    is_prime = np.ones(limit + 1, dtype=bool)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(limit ** 0.5) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = False
    primes = np.where(is_prime)[0]
    
    # indices for vectorization
    indices = np.arange(MAX_VAL + 1)
    
    for p in primes:
        p2 = p * p
        if p2 > MAX_VAL:
            break
        # We need to handle powers p^k for k >= 2
        pk = p2
        while pk <= MAX_VAL:
            # Multiples of pk
            # We want to multiply pp[m] by pk ONLY IF v_p(m) == log_p(pk)
            # i.e., m is divisible by pk but NOT by pk*p
            
            # Mask for multiples of pk
            mask = (indices % pk == 0)
            
            if pk * p <= MAX_VAL:
                # Mask for multiples of pk*p (these have v_p >= k+1)
                mask_next = (indices % (pk * p) == 0)
                # We want mask AND NOT mask_next
                mask = mask & ~mask_next
            else:
                # If pk*p > MAX_VAL, then no number <= MAX_VAL is divisible by pk*p
                # So all multiples of pk have exactly v_p = k (within our range)
                # Wait, if pk*p > MAX_VAL, then for any m <= MAX_VAL, if pk|m,
                # then v_p(m) cannot be >= k+1 because that would imply pk*p|m > MAX_VAL.
                # So yes, mask is just multiples of pk.
                pass
            
            pp[mask] *= pk
            pk *= p
            
            # Time check
            if time.time() - start_time > 85.0:
                # Fallback if something goes wrong, though this should be fast
                return 10000

    # Now find n in [10000, 1000000] maximizing pp[n] * pp[n+1]
    # Note: n can be 1000000, so we need pp[1000001].
    # Our pp array is size MAX_VAL+1 = 1000002, so indices 0..1000001 are valid.
    
    # Create array of products for n in [10000, 1000000]
    # n ranges from 10000 to 1000000 inclusive.
    # n+1 ranges from 10001 to 1000001.
    
    n_start = 10000
    n_end = 1000000
    
    # Slice pp for n and n+1
    pp_n = pp[n_start:n_end+1]
    pp_np1 = pp[n_start+1:n_end+2]
    
    products = pp_n * pp_np1
    
    # Find index of max
    max_idx = np.argmax(products)
    best_n = n_start + max_idx
    
    return int(best_n)
