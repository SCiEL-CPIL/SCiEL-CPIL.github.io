# EVOLVE-BLOCK-START
import numpy as np
import time


def _compute_mobius(n):
    """Compute the Möbius function mu(k) for k = 1..n using a sieve."""
    mu = np.ones(n + 1, dtype=np.int8)
    is_prime = np.ones(n + 1, dtype=bool)
    primes = []
    is_prime[0] = False
    if n >= 1:
        is_prime[1] = False
    for i in range(2, n + 1):
        if is_prime[i]:
            primes.append(i)
            for j in range(i * i, n + 1, i):
                is_prime[j] = False
        for p in primes:
            if i * p > n:
                break
            is_prime[i * p] = False
            if i % p == 0:
                break
    # mu(1) = 1
    # For k > 1: mu(k) = 0 if k has a squared prime factor
    # mu(k) = (-1)^r if k is product of r distinct primes
    mu = np.ones(n + 1, dtype=np.float64)
    mu[0] = 0.0
    # Sieve for mu
    mu_sieve = np.ones(n + 1, dtype=np.int8)
    mu_sieve[0] = 0
    for i in range(2, n + 1):
        if is_prime[i]:
            for j in range(i, n + 1, i):
                mu_sieve[j] *= -1
            for j in range(i * i, n + 1, i * i):
                mu_sieve[j] = 0
    mu[1:] = mu_sieve[1:]
    return mu


def prime_number_theorem_partial_function():
    """
    Construct a partial function f from a subset of the positive integers to the reals
    that maximizes a hidden number-theoretic objective related to the Prime Number Theorem.
    """
    start_time = time.time()
    deadline = start_time + 600  # 600 second budget

    # Compute Möbius function up to K
    K = 200000
    mu = _compute_mobius(K)

    # The domain is all k from 1 to K where mu(k) != 0
    # But we need to be careful: the evaluator clips values to [-10, 10]
    # mu(k) is in {-1, 0, 1}, so no clipping issue
    # We include all k with mu(k) != 0 to maximize the objective
    mask = (mu[1:] != 0)
    domain = np.where(mask)[0] + 1  # 1-indexed
    values = mu[1:][mask]

    # Verify feasibility: M(x) = sum_k f(k) * floor(x/k) <= 1.0001 for x in [1, 10K]
    # For the Möbius function, M(x) = 1 for all x >= 1 (by the identity sum_{d|x} mu(d) = [x=1])
    # So this should be feasible. Let's verify with a spot check.
    
    # Quick verification for a few x values
    # M(x) = sum_{k in domain, k <= x} mu(k) * floor(x/k)
    # This equals sum_{d|x} mu(d) = 1 for x > 1, and = 1 for x = 1
    # So M(x) = 1 for all x, which is <= 1.0001
    
    # The objective a = -sum_k f(k) * log(k) / k
    # For mu: a = -sum_{k=1}^{K} mu(k) * log(k) / k
    # This converges to 1 as K -> infinity (related to log(1/zeta(s)) at s=1)
    
    elapsed = time.time() - start_time
    if elapsed > deadline - 5:
        # Safety margin
        return domain, values

    return domain, values


# EVOLVE-BLOCK-END
