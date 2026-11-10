# EVOLVE-BLOCK-START
import numpy as np
import time


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
    deadline = start_time + 580  # safety margin before 630s limit

    # Fallback to seed result
    def fallback():
        domain = np.array([1, 2, 3, 5, 30], dtype=np.int64)
        values = np.array([3.47155351603266, -1.0, -1.0, -1.0, 1.0], dtype=np.float64)
        return domain, values

    # Try decreasing K values until feasible
    for K in [100000, 50000, 20000, 10000, 5000, 2000, 1000]:
        if time.time() > deadline:
            return fallback()

        # Compute Möbius function up to K using linear sieve
        mu = np.ones(K + 1, dtype=np.int64)
        mu[0] = 0
        is_prime = np.ones(K + 1, dtype=bool)
        is_prime[0] = is_prime[1] = False
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

        # Domain: k where mu[k] != 0, k >= 2 (exclude k=1 to avoid large f1*x term)
        mask = (mu != 0)
        mask[0] = False
        mask[1] = False
        domain = np.where(mask)[0]
        values = mu[domain].astype(np.float64)
        m = len(domain)

        if m < 1:
            continue

        # Compute S = sum f(k)/k over domain
        S = np.sum(values / domain)

        # f(1) = -S (since 1 not in domain, initial f(1)=0, then replaced by 0-S)
        f1 = -S

        # Compute M(x) for x in [1, 10K]
        max_x = 10 * K
        D = np.zeros(max_x + 2, dtype=np.float64)

        # For each k in domain, add fk to intervals where floor(x/k) is constant
        # floor(x/k) = j for x in [j*k, (j+1)*k - 1]
        for idx in range(m):
            k = int(domain[idx])
            fk = values[idx]
            j = 1
            while j * k <= max_x:
                lo = j * k
                hi = min((j + 1) * k - 1, max_x)
                D[lo] += fk
                if hi + 1 <= max_x:
                    D[hi + 1] -= fk
                j += 1

        # Prefix sum gives contribution from k >= 2
        # D has size max_x + 2, cumsum has size max_x + 2
        # We only need indices 0..max_x for M(x) where x in [0, max_x]
        M_k_geq_2 = np.cumsum(D)[:max_x + 1]

        # Add k=1 contribution: f1 * floor(x/1) = f1 * x
        # np.arange(max_x + 1) has size max_x + 1
        M = M_k_geq_2 + f1 * np.arange(max_x + 1, dtype=np.float64)
        M[0] = 0.0

        # Check constraint: M(x) <= 1.0001 for all x in [1, 10K]
        if np.max(M[1:]) > 1.0001:
            continue

        # Feasible! Compute objective (for verification, not needed for return)
        a = -np.sum(values * np.log(domain.astype(np.float64)) / domain)

        return domain, values

    return fallback()


# EVOLVE-BLOCK-END
