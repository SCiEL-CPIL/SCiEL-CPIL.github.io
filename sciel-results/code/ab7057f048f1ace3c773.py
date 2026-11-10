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
    deadline = 600  # seconds, safety margin before 630s kill

    def mobius_sieve(n):
        """Compute Mobius function mu(k) for k=1..n using a sieve."""
        mu = np.ones(n + 1, dtype=np.int8)
        mu[0] = 0
        is_prime = np.ones(n + 1, dtype=bool)
        is_prime[0] = is_prime[1] = False
        primes = []
        for i in range(2, n + 1):
            if is_prime[i]:
                primes.append(i)
                mu[i] = -1
            for p in primes:
                if i * p > n:
                    break
                is_prime[i * p] = False
                if i % p == 0:
                    mu[i * p] = 0
                    break
                else:
                    mu[i * p] = -mu[i]
        return mu

    # Try increasing K values and pick the best feasible one
    best_domain = None
    best_values = None
    best_score = -np.inf

    for K in [1000, 5000, 10000, 20000, 50000, 100000, 200000]:
        if time.time() - start_time > deadline - 30:
            break

        mu = mobius_sieve(K)
        # Get indices where mu != 0
        nonzero_idx = np.where(mu[1:] != 0)[0] + 1
        domain = nonzero_idx.astype(np.int64)
        values = mu[nonzero_idx].astype(np.float64)

        m = len(domain)
        if m < 1:
            continue

        # Check time: evaluator does m * 10*K operations
        est_ops = m * 10 * K
        if est_ops > 2e9:  # ~10-20 seconds for evaluator
            continue

        # Compute the objective a = -sum f(k) log(k) / k
        log_k = np.log(domain.astype(np.float64))
        a = -np.sum(values * log_k / domain.astype(np.float64))

        # Verify feasibility: M(x) = sum_k f(k) * floor(x/k) <= 1.0001 for x in [1, 10K]
        # We can check a sample of x values to save time
        max_x = 10 * K
        # Check x = 1, 2, ..., min(max_x, 1000) and some larger values
        check_xs = list(range(1, min(max_x, 1000) + 1))
        if max_x > 1000:
            # Add some larger values
            extra = np.linspace(1000, max_x, min(100, max_x - 1000)).astype(int)
            check_xs.extend(extra.tolist())
        check_xs = sorted(set(check_xs))

        feasible = True
        for x in check_xs:
            Mx = np.sum(values * (x // domain))
            if Mx > 1.0001:
                feasible = False
                break

        if not feasible:
            continue

        if a > best_score:
            best_score = a
            best_domain = domain.copy()
            best_values = values.copy()

    # Fallback to baseline if nothing found
    if best_domain is None:
        domain = np.array([1, 2, 3, 5, 30], dtype=np.int64)
        values = np.array([3.47155351603266, -1.0, -1.0, -1.0, 1.0], dtype=np.float64)
    else:
        domain = best_domain
        values = best_values

    return domain, values


# EVOLVE-BLOCK-END
