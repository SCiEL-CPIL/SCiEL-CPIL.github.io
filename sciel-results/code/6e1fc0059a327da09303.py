# EVOLVE-BLOCK-START
import numpy as np
import time


def prime_number_theorem_partial_function():
    """
    Construct a partial function f from a subset of the positive integers to the reals
    that maximizes a hidden number-theoretic objective related to the Prime Number Theorem.
    """
    start_time = time.time()
    deadline = start_time + 600  # safety margin before 630s kill

    # Compute Möbius function up to K
    K = 200000
    mu = np.ones(K + 1, dtype=np.int64)
    primes = []
    is_prime = np.ones(K + 1, dtype=bool)
    is_prime[0] = False
    is_prime[1] = False

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

    # Filter to non-zero mu values
    mask = mu[1:] != 0
    domain = np.where(mask)[0] + 1
    values = mu[1:][mask].astype(np.float64)

    # Check time
    if time.time() > deadline:
        # Return a small feasible subset as fallback
        domain = np.array([1, 2, 3, 5, 30], dtype=np.int64)
        values = np.array([3.47155351603266, -1.0, -1.0, -1.0, 1.0], dtype=np.float64)
        return domain, values

    return domain, values


# EVOLVE-BLOCK-END
