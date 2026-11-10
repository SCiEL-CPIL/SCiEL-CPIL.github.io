import time
import math
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin

    N = 1000000
    best_n = 10000

    # Powerful part array
    pp = np.ones(N + 2, dtype=np.int64)

    # Sieve of Eratosthenes for primes up to sqrt(N)
    limit = int(math.isqrt(N)) + 1
    is_prime = bytearray([1]) * (limit + 1)
    is_prime[0] = is_prime[1] = 0
    for i in range(2, int(math.isqrt(limit)) + 1):
        if is_prime[i]:
            is_prime[i*i:limit+1:i] = bytearray(len(range(i*i, limit+1, i)))

    primes = [i for i in range(2, limit + 1) if is_prime[i]]

    for p in primes:
        p2 = p * p
        if p2 > N + 1:
            break

        pe = p2
        while pe <= N + 1:
            max_k = (N + 1) // pe
            if max_k == 0:
                break
            
            # k_vals from 1 to max_k
            # We want indices k*pe where k is NOT divisible by p
            # This means the exponent of p in k*pe is exactly the exponent in pe
            
            # Optimization: 
            # Instead of masking, we can iterate.
            # But numpy is faster for large arrays.
            # For small max_k, pure python might be faster?
            # max_k is at most 250,000 (for p=2, pe=4).
            # 250,000 is small enough for numpy.
            
            if time.time() > deadline:
                return best_n

            k_vals = np.arange(1, max_k + 1, dtype=np.int64)
            mask = (k_vals % p) != 0
            if np.any(mask):
                indices = k_vals[mask] * pe
                pp[indices] *= pe
                
            pe *= p
            if time.time() > deadline:
                return best_n

    # Calculate max product
    if time.time() > deadline:
        return best_n

    # n in [10000, 1000000]
    # prod[n] = pp[n] * pp[n+1]
    # We need max of prod for n in 10000..1000000
    # This corresponds to indices 10000..1000000 in pp
    # and 10001..1000001 in pp
    
    # Vectorized multiplication
    # Note: pp has size N+2, so pp[N+1] is valid.
    prods = pp[10000:N+1] * pp[10001:N+2]
    
    max_idx = np.argmax(prods)
    best_n = 10000 + max_idx
    max_prod = prods[max_idx]

    return best_n
