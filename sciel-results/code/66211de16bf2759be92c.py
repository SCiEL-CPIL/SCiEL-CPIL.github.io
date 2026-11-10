import numpy as np
import time


def prime_number_theorem_partial_function():
    start_time = time.time()
    deadline = start_time + 580

    max_K = 200000

    # Linear sieve for Möbius function
    mu = np.zeros(max_K + 1, dtype=np.int64)
    mu[1] = 1
    is_prime = np.ones(max_K + 1, dtype=bool)
    is_prime[0] = is_prime[1] = False
    primes = []
    for i in range(2, max_K + 1):
        if is_prime[i]:
            primes.append(i)
            mu[i] = -1
        for p in primes:
            if i * p > max_K:
                break
            is_prime[i * p] = False
            if i % p == 0:
                mu[i * p] = 0
                break
            else:
                mu[i * p] = -mu[i]

    candidate_Ks = [200000, 150000, 100000, 75000, 50000, 30000, 20000, 10000, 5000, 2000, 1000, 500, 200, 100, 50, 20, 10, 5]

    for K in candidate_Ks:
        if time.time() > deadline:
            break

        max_x = 10 * K
        diff = np.zeros(max_x + 2, dtype=np.float64)

        for k in range(1, K + 1):
            mk = mu[k]
            if mk == 0:
                continue
            val = float(mk)
            j_max = max_x // k
            if j_max == 0:
                continue
            positions = np.arange(1, j_max + 1, dtype=np.int64) * k
            diff[positions] += val
            next_pos = positions + k
            valid = next_pos <= max_x
            if np.any(valid):
                diff[next_pos[valid]] -= val

        M = np.cumsum(diff)

        if np.max(M[1:max_x + 1]) <= 1.0001:
            domain = np.arange(1, K + 1, dtype=np.int64)
            values = mu[1:K + 1].astype(np.float64)
            return domain, values

    # Fallback
    domain = np.array([1, 2, 3, 5, 30], dtype=np.int64)
    values = np.array([3.47155351603266, -1.0, -1.0, -1.0, 1.0], dtype=np.float64)
    return domain, values
