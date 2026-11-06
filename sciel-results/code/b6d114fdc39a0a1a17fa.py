import time
import numpy as np

def construct():
    t0 = time.time()
    DEADLINE = 85  # safety margin before 90s kill

    def elapsed():
        return time.time() - t0

    MAXN = 1_000_000
    
    # Initialize powerful part array with 1s
    pp = np.ones(MAXN + 2, dtype=np.int64)
    
    # Sieve for primes up to sqrt(MAXN)
    # We only need primes p such that p^2 <= MAXN
    limit = int(MAXN**0.5) + 1
    is_prime = np.ones(limit + 1, dtype=bool)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(limit**0.5) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = False
    primes = np.where(is_prime)[0]
    
    # Calculate powerful parts
    for p in primes:
        if elapsed() > DEADLINE:
            break
            
        p_pow = p * p
        j = 2
        while p_pow <= MAXN:
            if elapsed() > DEADLINE:
                break
            
            # Identify indices m such that v_p(m) == j
            # These are multiples of p^j that are NOT multiples of p^(j+1)
            # m = k * p^j where k is not divisible by p
            
            # Vectorized approach:
            # Get all multiples of p^j
            indices = np.arange(p_pow, MAXN + 1, p_pow)
            
            # Get the quotients k = m / p^j
            quotients = indices // p_pow
            
            # Mask out quotients divisible by p
            mask = (quotients % p) != 0
            
            # Apply the powerful part factor p^j to the valid indices
            if np.any(mask):
                pp[indices[mask]] *= p_pow
            
            p_pow *= p
            j += 1

    if elapsed() > DEADLINE:
        return 10000
    
    # Scan for the best pair
    start = 10000
    # n ranges from start to MAXN-1
    # We need pp[n] * pp[n+1]
    
    products = pp[start:MAXN] * pp[start+1:MAXN+1]
    
    max_idx = np.argmax(products)
    best_val = products[max_idx]
    best_n = start + max_idx
    
    return int(best_n)
