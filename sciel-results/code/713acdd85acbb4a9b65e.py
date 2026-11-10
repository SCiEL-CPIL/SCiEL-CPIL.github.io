# EVOLVE-BLOCK-START
import numpy as np
import time


def prime_number_theorem_partial_function():
    """
    Construct f(k) = mu(k) (Möbius function) for k = 2..K, with f(1) = 0.
    
    The Möbius function satisfies the exact identity:
        M(x) = sum_{k<=x} mu(k) * floor(x/k) = 1  for all x >= 1
    
    This makes the construction exactly feasible (M(x) = 1 <= 1.0001).
    The objective a = -sum_k f(k)*log(k)/k approaches 1 as K -> infinity
    (Euler product: -sum mu(k) log(k)/k = log(zeta(1)) -> 1).
    
    After the evaluator's correction (f(1) -= S where S = sum f(k)/k),
    the sum f(k)/k becomes 0, and the objective is maximized.
    """
    deadline = time.time() + 600.0
    K = 200000
    
    # Sieve for Möbius function up to K
    mu = np.ones(K + 1, dtype=np.int8)
    is_prime = np.ones(K + 1, dtype=bool)
    is_prime[0] = False
    is_prime[1] = False
    primes = []
    
    for i in range(2, K + 1):
        if is_prime[i]:
            primes.append(i)
            mu[i] = -1
            i2 = i * i
            if i2 <= K:
                # Mark multiples of i^2 as having square factor
                for j in range(i2, K + 1, i2):
                    mu[j] = 0
                # Mark composites
                for j in range(i2, K + 1, i):
                    is_prime[j] = False
    
    # Proper Möbius sieve: for each prime p, multiply mu by -1 for multiples
    # Reset and do proper linear sieve
    mu = np.ones(K + 1, dtype=np.int64)
    is_prime = np.ones(K + 1, dtype=bool)
    is_prime[0] = False
    is_prime[1] = False
    primes = []
    min_prime_factor = np.zeros(K + 1, dtype=np.int64)
    
    for i in range(2, K + 1):
        if is_prime[i]:
            primes.append(i)
            min_prime_factor[i] = i
        for p in primes:
            if p > min_prime_factor[i] or i * p > K:
                break
            is_prime[i * p] = False
            min_prime_factor[i * p] = p
            if i % p == 0:
                mu[i * p] = 0
                break
            else:
                mu[i * p] = -mu[i]
    
    # Build domain and values: k = 2..K, f(k) = mu(k), f(1) = 0
    # Domain: [1, 2, 3, ..., K]
    domain = np.arange(1, K + 1, dtype=np.int64)
    
    # Values: f(1) = 0, f(k) = mu(k) for k >= 2
    values = np.zeros(K, dtype=np.float64)
    values[0] = 0.0  # f(1)
    values[1:] = mu[2:K+1].astype(np.float64)
    
    # Clip to [-10, 10] as the evaluator will (mu values are -1, 0, 1 so no effect)
    values = np.clip(values, -10.0, 10.0)
    
    return domain, values


# EVOLVE-BLOCK-END
