# EVOLVE-BLOCK-START
import numpy as np


def prime_number_theorem_partial_function():
    """
    Construct a partial function f from a subset of the positive integers to the reals
    that maximizes a hidden number-theoretic objective related to the Prime Number Theorem.

    Strategy:
      Use the exact Mobius function mu(k) for all square-free k <= K.
      The infinite identity sum_{k=1}^infty mu(k)/k = 0 and
      M(x) = sum_{k=1}^x mu(k) floor(x/k) = 1 for all x >= 1 (in the limit)
      suggests that truncating to finite K gives a near-optimal feasible solution.

      For finite K, sum_{k<=K} mu(k)/k is not exactly 0, but the evaluator
      corrects f(1) so that sum f(k)/k = 0 afterwards. The constraint M(x) <= 1.0001
      must hold for all x in [1, 10K].

      We compute mu(k) via sieve for k up to K, keep only square-free indices,
      and set values = mu(k). Then the evaluator clips values to [-10,10] (no effect
      since mu is in {-1,0,1}), adjusts f(1), and checks feasibility.

      The objective a = -sum f(k) log(k)/k should be close to 1 for large K.
    """
    # Maximum key K
    K = 200000

    # Compute Mobius function up to K using a sieve
    mu = np.ones(K + 1, dtype=np.int8)
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

    # mu[1] = 1 by definition
    mu[1] = 1

    # Keep only indices where mu(k) != 0 (square-free integers)
    mask = mu != 0
    domain = np.where(mask)[0] + 1  # 1-indexed
    values = mu[domain].astype(np.float64)

    return domain, values


# EVOLVE-BLOCK-END
