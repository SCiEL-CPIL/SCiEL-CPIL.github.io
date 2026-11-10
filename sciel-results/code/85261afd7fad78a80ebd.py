import numpy as np
import time


def prime_number_theorem_partial_function():
    """
    Construct a partial function f based on the Möbius function to maximize
    the prime number theorem objective.
    
    Returns:
        (domain, values): tuple of two numpy arrays
    """
    K = 200000
    
    # Linear sieve for Möbius function
    mu = np.ones(K + 1, dtype=np.int64)
    mu[0] = 0
    primes = []
    is_composite = np.zeros(K + 1, dtype=bool)
    
    for i in range(2, K + 1):
        if not is_composite[i]:
            primes.append(i)
        for p in primes:
            if i * p > K:
                break
            is_composite[i * p] = True
            if i % p == 0:
                mu[i * p] = 0
                break
            else:
                mu[i * p] = -mu[i]
    
    # Build domain and values: include only k where mu(k) != 0
    mask = mu[1:] != 0
    domain = np.where(mask)[0] + 1
    values = mu[domain].astype(np.float64)
    
    return domain, values
