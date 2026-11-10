# EVOLVE-BLOCK-START
import numpy as np
import time


def prime_number_theorem_partial_function():
    start_time = time.time()
    DEADLINE = 620.0
    
    K_max = 200000
    mu = np.zeros(K_max + 1, dtype=np.int8)
    mu[1] = 1
    primes = []
    is_composite = np.zeros(K_max + 1, dtype=bool)
    
    for i in range(2, K_max + 1):
        if not is_composite[i]:
            primes.append(i)
            mu[i] = -1
        for p in primes:
            if i * p > K_max:
                break
            is_composite[i * p] = True
            if i % p == 0:
                mu[i * p] = 0
                break
            else:
                mu[i * p] = -mu[i]
    
    domain_mask = mu[1:] != 0
    domain = np.where(domain_mask)[0] + 1
    values = mu[domain].astype(np.float64)
    
    X_max = 10 * K_max
    M = np.zeros(X_max + 1, dtype=np.float64)
    
    for idx in range(len(domain)):
        k = domain[idx]
        f_k = values[idx]
        for start in range(k, X_max + 1, k):
            end = min(start + k, X_max + 1)
            M[start:end] += f_k
    
    max_M = np.max(M[1:])
    
    if max_M > 1.0001:
        scale = 1.0001 / max_M
        values = values * scale
    
    return domain, values

# EVOLVE-BLOCK-END
