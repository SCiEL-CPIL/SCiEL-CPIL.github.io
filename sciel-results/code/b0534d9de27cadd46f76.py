import time
import numpy as np
from math import isqrt

def construct():
    start_time = time.time()
    deadline = start_time + 85

    N = 1000001

    # 1. Generate Primes
    is_prime = np.ones(N + 1, dtype=bool)
    is_prime[0] = False
    is_prime[1] = False
    for i in range(2, isqrt(N) + 1):
        if is_prime[i]:
            is_prime[i*i:N+1:i] = False
    
    primes = np.where(is_prime)[0]

    # 2. Compute Powerful Part P
    P = np.ones(N + 1, dtype=np.int64)
    
    for p in primes:
        p2 = p * p
        if p2 > N:
            break
        
        pk = p2
        while pk <= N:
            p_next = pk * p
            # Multiples of pk
            multiples = np.arange(pk, N + 1, pk)
            # Filter those not divisible by p_next
            mask = (multiples % p_next) != 0
            if np.any(mask):
                P[multiples[mask]] *= pk
            pk = p_next
            
    # 3. Find Max Product
    # n in [10000, 1000000]
    # P[n] * P[n+1]
    # P is 0-indexed.
    # n corresponds to P[n]
    # We want max of P[n] * P[n+1] for n in [10000, 1000000]
    
    scores = P[10000:1000001] * P[10001:1000002]
    
    best_idx = np.argmax(scores)
    best_n = 10000 + best_idx
    
    return int(best_n)
