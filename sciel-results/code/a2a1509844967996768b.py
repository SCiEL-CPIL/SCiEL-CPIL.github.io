# EVOLVE-BLOCK-START
import numpy as np
import time

def prime_number_theorem_partial_function():
    start_time = time.time()
    deadline = start_time + 600  # Safety margin
    
    # Compute Möbius function up to K
    K = 100000
    mu = np.ones(K + 1, dtype=np.int64)
    primes = []
    is_prime = np.ones(K + 1, dtype=bool)
    is_prime[0] = is_prime[1] = False
    
    for i in range(2, K + 1):
        if is_prime[i]:
            primes.append(i)
            for j in range(i, K + 1, i):
                is_prime[j] = False
                mu[j] *= -1
            i2 = i * i
            if i2 <= K:
                for j in range(i2, K + 1, i2):
                    mu[j] = 0
    
    # Filter domain to where mu(k) != 0
    domain = np.array([i for i in range(1, K + 1) if mu[i] != 0], dtype=np.int64)
    values = mu[domain].astype(np.float64)
    
    # Check time
    if time.time() > deadline:
        # Fallback to a small hardcoded solution
        domain = np.array([1, 2, 3, 5, 30], dtype=np.int64)
        values = np.array([3.47155351603266, -1.0, -1.0, -1.0, 1.0], dtype=np.float64)
        return domain, values
    
    return domain, values

# EVOLVE-BLOCK-END
