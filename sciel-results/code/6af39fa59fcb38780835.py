# EVOLVE-BLOCK-START
import numpy as np
import time


def prime_number_theorem_partial_function():
    """
    Compute the Mobius function mu(k) for k = 1..K, which is the optimal
    partial function for the Prime Number Theorem objective.

    The Mobius function satisfies:
      - sum_{k<=x} mu(k) * floor(x/k) = 1 for all x >= 1 (feasibility)
      - -sum_{k=2}^K mu(k) * log(k) / k -> 1 as K -> infinity (optimal metric)
    """
    start_time = time.time()
    DEADLINE = 600.0  # safety margin before 630s kill

    # Compute Mobius function up to K using a linear sieve
    K = 200000

    # Linear sieve for Mobius function
    mu = np.ones(K + 1, dtype=np.int8)
    primes = []
    is_composite = np.zeros(K + 1, dtype=bool)
    mu[0] = 0
    mu[1] = 1

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

    # Check deadline
    if time.time() - start_time > DEADLINE - 30:
        # Return a smaller domain as fallback
        K = 10000
        mu = np.ones(K + 1, dtype=np.int8)
        primes = []
        is_composite = np.zeros(K + 1, dtype=bool)
        mu[0] = 0
        mu[1] = 1
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

    # Build domain: all k from 1 to K where mu(k) != 0
    # Actually, we should include all k since mu(k) can be 0, -1, or 1
    # Including mu(k)=0 terms doesn't hurt (they contribute 0 to sums)
    # but to keep domain smaller, we can exclude them
    # However, the evaluator computes M(x) = sum_k f(k)*floor(x/k) over all k in domain
    # If we exclude mu(k)=0 terms, M(x) is unchanged since they contribute 0
    # Let's include all k=1..K for correctness and simplicity

    domain = np.arange(1, K + 1, dtype=np.int64)
    values = mu[1:K + 1].astype(np.float64)

    return domain, values


# EVOLVE-BLOCK-END
