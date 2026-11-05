import time
import math
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin before 90s limit
    N = 1000000
    MIN_N = 10000
    
    # 1. Sieve of Eratosthenes to get primes up to sqrt(N)
    sqrt_N = int(math.isqrt(N))
    is_prime = np.ones(sqrt_N + 1, dtype=bool)
    is_prime[0] = False
    is_prime[1] = False
    for i in range(2, int(math.isqrt(sqrt_N)) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = False
    
    primes = np.where(is_prime)[0]
    
    # 2. Compute powerful part P(n) for all n in [1, N] using multiplicative sieve with NumPy
    # P[n] will store the powerful part of n
    P = np.ones(N + 1, dtype=np.int64)
    
    for p in primes:
        p_int = int(p)
        pk = p_int * p_int
        k = 2
        while pk <= N:
            limit = N // pk
            if limit >= 1:
                # Generate all multiples of pk: pk, 2pk, ..., limit*pk
                # We need to multiply P[m*pk] by pk ONLY IF m is NOT divisible by p
                # m ranges from 1 to limit
                m_vals = np.arange(1, limit + 1)
                # Mask for m not divisible by p
                mask = (m_vals % p_int) != 0
                # Get indices in P array
                # indices = m_vals[mask] * pk
                # To avoid creating a huge intermediate array if possible, 
                # but limit is at most 10^6/4 = 250k, which is fine.
                indices = m_vals[mask] * pk
                P[indices] *= pk
            
            pk *= p_int
            k += 1
            
            if time.time() > deadline - 10:
                break
        
        if time.time() > deadline - 10:
            break

    # 3. Find max P[n] * P[n+1] for n in [10000, N-1]
    # We want n such that n in [10000, N-1]
    # So n+1 in [10001, N]
    # Slice P from 10000 to N-1 (exclusive) for first term
    # Slice P from 10001 to N (exclusive) for second term?
    # P[10000] * P[10001]
    # P[10001] * P[10002]
    # ...
    # P[N-1] * P[N]
    
    # Array of products
    # First part: P[10000 : N]  (length N - 10000)
    # Second part: P[10001 : N+1] (length N - 10000)
    
    # Note: N is 1000000.
    # P indices 0..N
    # We need n up to N-1 because n+1 must be <= N.
    # So n in [10000, N-1]
    
    p_n = P[MIN_N : N]       # P[10000] ... P[N-1]
    p_np1 = P[MIN_N + 1 : N + 1] # P[10001] ... P[N]
    
    products = p_n * p_np1
    
    best_idx = np.argmax(products)
    best_n = MIN_N + best_idx
    
    # Validate
    if best_n < MIN_N or best_n > N - 1:
        best_n = MIN_N
        
    return int(best_n)
