# EVOLVE-BLOCK-START
import numpy as np
import time


def _compute_mobius(K):
    """Compute the Möbius function mu(k) for k = 1..K using a sieve."""
    mu = np.ones(K + 1, dtype=np.float64)
    mu[0] = 0.0
    if K >= 1:
        mu[1] = 1.0
    
    # Sieve of Eratosthenes variant for Möbius
    is_prime = np.ones(K + 1, dtype=bool)
    is_prime[0] = False
    if K >= 1:
        is_prime[1] = False
    
    for i in range(2, K + 1):
        if is_prime[i]:
            # i is prime
            for j in range(i, K + 1, i):
                is_prime[j] = False
            mu[i] = -1.0
            i2 = i * i
            for j in range(i2, K + 1, i2):
                mu[j] = 0.0
            for j in range(i * 2, K + 1, i):
                if j % i2 != 0:
                    mu[j] *= -1.0
    
    # Fix: the above sieve is a bit off. Let me use a cleaner approach.
    mu = np.ones(K + 1, dtype=np.float64)
    mu[0] = 0.0
    if K >= 1:
        mu[1] = 1.0
    
    primes = []
    is_prime = np.ones(K + 1, dtype=bool)
    is_prime[0] = False
    if K >= 1:
        is_prime[1] = False
    
    for i in range(2, K + 1):
        if is_prime[i]:
            primes.append(i)
            for j in range(i * i, K + 1, i):
                is_prime[j] = False
        for p in primes:
            if i * p > K:
                break
            is_prime[i * p] = False
            if i % p == 0:
                mu[i * p] = 0.0
                break
            else:
                mu[i * p] = -mu[i]
    
    return mu


def prime_number_theorem_partial_function():
    """
    Construct a partial function f from a subset of the positive integers to the reals
    that maximizes a hidden number-theoretic objective related to the Prime Number Theorem.
    """
    start_time = time.time()
    deadline = start_time + 570  # safety margin before 630s kill
    
    # Use the Möbius function. For large K, a = -sum mu(k)*log(k)/k approaches 1.
    # The constraint M(x) <= 1.0001 is satisfied by the Möbius function for reasonable K.
    
    K = 50000
    
    mu = _compute_mobius(K)
    
    # Build domain: all k where mu(k) != 0
    mask = mu[1:] != 0
    domain = np.where(mask)[0] + 1
    values = mu[domain]
    
    elapsed = time.time() - start_time
    if elapsed > 500:
        return domain, values
    
    # Verify feasibility at critical points
    max_x = 10 * K
    m = len(domain)
    
    # M(x) = sum values[i] * floor(x / domain[i])
    # Check at x = 1, and at domain points, and at max_x
    test_points = np.unique(np.concatenate([domain, np.array([1, max_x])]))
    
    feasible = True
    for x in test_points:
        Mx = np.dot(values, x // domain)
        if Mx > 1.0001:
            feasible = False
            break
    
    if not feasible:
        # Reduce K
        K = 10000
        mu = _compute_mobius(K)
        mask = mu[1:] != 0
        domain = np.where(mask)[0] + 1
        values = mu[domain]
    
    return domain, values


# EVOLVE-BLOCK-END
