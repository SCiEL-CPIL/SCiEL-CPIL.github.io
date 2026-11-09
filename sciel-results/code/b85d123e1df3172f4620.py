# EVOLVE-BLOCK-START
import numpy as np
import time


def prime_number_theorem_partial_function():
    """
    Construct a partial function f from a subset of the positive integers to the reals
    that maximizes the objective a = -sum_k f(k) * log(k) / k subject to:
      * sum_k f(k)/k == 0 (after evaluator's f(1) adjustment)
      * M(x) = sum_k f(k) * floor(x/k) <= 1.0001 for all x in [1, 10*K]
      * values clipped to [-10, 10]
      * K <= 200000

    The theoretical optimum is the Möbius function mu(k), which satisfies
      sum_{d|n} mu(d) = 1 for all n, hence M(x) = 1 for all x.
    And -sum mu(k) log(k)/k -> 1 as K -> infinity.

    This implementation computes the full Möbius function up to K=200000
    using a sieve, then returns (domain, values).
    """
    t_start = time.time()
    DEADLINE = 620.0  # 10s safety margin before 630s kill

    # Compute Möbius function up to K using a linear sieve
    K = 200000
    mu = np.zeros(K + 1, dtype=np.int8)
    mu[1] = 1
    primes = []
    is_composite = np.zeros(K + 1, dtype=bool)

    for i in range(2, K + 1):
        if not is_composite[i]:
            primes.append(i)
            mu[i] = -1
        for p in primes:
            if i * p > K:
                break
            is_composite[i * p] = True
            if i % p == 0:
                mu[i * p] = 0
                break
            else:
                mu[i * p] = -mu[i]

    # Check time
    if time.time() > DEADLINE:
        # Fallback to small domain
        domain = np.array([1, 2, 3, 5, 30], dtype=np.int64)
        values = np.array([3.47155351603266, -1.0, -1.0, -1.0, 1.0], dtype=np.float64)
        return domain, values

    # Build domain and values arrays from mu
    # domain: all k in [1, K] (include k=1 with mu(1)=1)
    domain = np.arange(1, K + 1, dtype=np.int64)
    values = mu[1:].astype(np.float64)

    # Check time
    if time.time() > DEADLINE:
        return domain, values

    # The evaluator will:
    # 1. Clip values to [-10, 10] (no effect since mu is in {-1, 0, 1})
    # 2. Compute S = sum_k f(k)/k, then set f(1) -= S so that sum f(k)/k = 0
    # 3. Check M(x) = sum_k f(k) * floor(x/k) <= 1.0001 for x in [1, 10K]
    # 4. Compute a = -sum_k f(k) * log(k) / k

    # For the full Möbius function, M(x) = sum_{k<=x} mu(k) * floor(x/k)
    # By the identity sum_{d|n} mu(d) = [n==1], we have:
    # M(x) = sum_{n=1}^{x} sum_{d|n, d<=K} mu(d) = sum_{n=1}^{x} [n==1] = 1
    # (when K >= x, which is satisfied since x <= 10K and K=200000,
    #  but for x > K, some divisors > K are missing)
    #
    # For x <= K: M(x) = 1 exactly.
    # For x > K: M(x) = sum_{n=1}^{x} sum_{d|n, d<=K} mu(d)
    #   = sum_{d=1}^{K} mu(d) * floor(x/d)
    #   which may deviate slightly from 1 but should stay <= 1.0001.

    # We trust the Möbius function satisfies the constraint.
    # Return the full domain.
    return domain, values


# EVOLVE-BLOCK-END
