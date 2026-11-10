# EVOLVE-BLOCK-START
import numpy as np
import time


def _compute_mobius(K):
    """Compute the Mobius function mu(n) for n=1..K using a sieve."""
    mu = np.ones(K + 1, dtype=np.int64)
    is_prime = np.ones(K + 1, dtype=bool)
    primes = []
    for i in range(2, K + 1):
        if is_prime[i]:
            primes.append(i)
            for p in primes:
                if i * p > K:
                    break
                is_prime[i * p] = False
                if i % p == 0:
                    mu[i * p] = 0
                    break
                else:
                    mu[i * p] = -mu[i]
    return mu


def prime_number_theorem_partial_function():
    """
    Construct a partial function f from a subset of the positive integers to the reals
    that maximizes a hidden number-theoretic objective related to the Prime Number Theorem.

    The function f is scored by a fixed evaluator (see evaluator.py) that:
      * subtracts a constant from f(1) so that sum_k f(k)/k == 0,
      * requires the step function M(x) = sum_k f(k) * floor(x/k) to stay <= 1 for every
        real x in [1, 10*max_key] (otherwise the construction is infeasible), and
      * returns the objective a = -sum_k f(k) * log(k) / k (to be maximized; optimum 1.0).
    The optimum is approached by the Mobius function mu(k), for which the classic identity
    sum_{k<=x} mu(k) floor(x/k) = 1 holds and -sum mu(k) log(k)/k -> 1.

    Representation:
        domain : np.ndarray, shape (m,), dtype int   -> positive integers in the domain of f
        values : np.ndarray, shape (m,), dtype float -> the corresponding real values f(k)

    Returns:
        (domain, values): a tuple of two numpy arrays of equal length describing f.
    """
    start_time = time.time()
    deadline = 600.0  # safety margin

    # Try increasing K until we run out of time
    best_result = None
    best_a = -np.inf

    # Start with a reasonable K and increase
    for K in [5000, 10000, 20000, 50000, 100000, 200000]:
        if time.time() - start_time > deadline - 60:
            break

        # Compute Mobius function
        mu = _compute_mobius(K)

        # Only include non-zero entries to keep m small
        mask = mu[1:K+1] != 0
        domain = np.where(mask)[0] + 1  # 1-indexed
        values = mu[domain].astype(np.float64)

        if len(domain) == 0:
            continue

        # Compute M(x) for x in [1, 10K] using difference array
        max_x = 10 * K
        diff = np.zeros(max_x + 2, dtype=np.float64)

        for i in range(len(domain)):
            k = int(domain[i])
            f_k = values[i]
            # floor(x/k) = j for x in [j*k, min((j+1)*k - 1, max_x)]
            for j in range(1, max_x // k + 1):
                start = j * k
                end = min((j + 1) * k - 1, max_x)
                if start > max_x:
                    break
                diff[start] += f_k * j
                diff[end + 1] -= f_k * j

        # Prefix sum to get M
        M = np.cumsum(diff)
        M = M[:max_x + 1]

        # Check constraint: M(x) <= 1.0001 for all x in [1, 10K]
        if np.max(M[1:max_x+1]) > 1.0001:
            continue

        # Compute objective a = -sum_k f(k) * log(k) / k
        k_float = domain.astype(np.float64)
        a = -np.sum(values * np.log(k_float) / k_float)

        if a > best_a:
            best_a = a
            best_result = (domain, values)

        if time.time() - start_time > deadline - 30:
            break

    if best_result is None:
        # Fallback to the seed
        domain = np.array([1, 2, 3, 5, 30], dtype=np.int64)
        values = np.array([3.47155351603266, -1.0, -1.0, -1.0, 1.0], dtype=np.float64)
        return domain, values

    return best_result


# EVOLVE-BLOCK-END
