# EVOLVE-BLOCK-START
import numpy as np
import time

def prime_number_theorem_partial_function():
    start_time = time.time()
    K = 200000
    
    # Compute Mobius function
    mu = np.ones(K + 1, dtype=np.int64)
    is_prime = np.ones(K + 1, dtype=bool)
    primes = []
    for i in range(2, K + 1):
        if is_prime[i]:
            primes.append(i)
            mu[i] = -1
        for p in primes:
            if i * p > K:
                break
            is_prime[i * p] = False
            if i % p == 0:
                mu[i * p] = 0
                break
            else:
                mu[i * p] = -mu[i]
                
    domain = np.arange(1, K + 1, dtype=np.int64)
    values = mu[1:].astype(np.float64)
    
    # Adjust f(1)
    k_arr = np.arange(1, K + 1, dtype=np.float64)
    S = np.sum(values / k_arr)
    values_adj = values.copy()
    values_adj[0] = values_adj[0] - S
    
    # Check M(x) for a few points
    # M(x) = sum values_adj[k-1] * (x // k)
    
    # Check x = 1
    M_1 = np.sum(values_adj) # floor(1/k) is 1 for k=1, 0 else.
    # Wait, floor(1/1)=1, floor(1/2)=0.
    # So M(1) = f(1).
    # f(1) is adjusted.
    
    # Check x = K
    # M(K) = sum_{k=1}^K f(k) floor(K/k)
    
    # If M(1) > 1.0001 or M(K) > 1.0001, it's bad.
    
    # Actually, M(1) = f(1).
    # f(1) = 1 - S.
    # S = sum_{k=2}^K mu(k)/k.
    # S is small.
    # f(1) is close to 1.
    
    # I will return the values.
    
    return domain, values

# EVOLVE-BLOCK-END
