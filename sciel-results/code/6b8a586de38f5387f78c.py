# EVOLVE-BLOCK-START
import time
import numpy as np


def _mobius_up_to(n):
    """Compute the Mobius function mu(k) for k = 1..n using a sieve."""
    mu = np.ones(n + 1, dtype=np.float64)
    is_prime = np.ones(n + 1, dtype=bool)
    primes = []
    for i in range(2, n + 1):
        if is_prime[i]:
            primes.append(i)
            mu[i] = -1.0
        for p in primes:
            if i * p > n:
                break
            is_prime[i * p] = False
            if i % p == 0:
                mu[i * p] = 0.0
                break
            else:
                mu[i * p] = -mu[i]
    return mu


def _compute_M_max(domain, values, K):
    """Compute max of M(x) = sum_k f(k)*floor(x/k) for x in [1, 10K]."""
    m = len(domain)
    if m == 0:
        return 0.0
    x_max = 10 * K
    # M(x) is a step function; it only changes at integer x. We evaluate at all integers.
    # To be efficient, note M(x) = sum_k f(k) * floor(x/k).
    # We can compute for x = 1..x_max.
    # Use vectorized approach: for each k, floor(x/k) is constant on intervals.
    # Simpler: build an array M of length x_max+1 (index = x).
    M = np.zeros(x_max + 1, dtype=np.float64)
    for k, f in zip(domain, values):
        if f == 0.0:
            continue
        k = int(k)
        # floor(x/k) for x = 1..x_max
        # We add f * floor(x/k) to M[x]
        # floor(x/k) takes values 0,1,2,... up to x_max//k
        # For x in [j*k, (j+1)*k - 1], floor(x/k) = j
        # We can accumulate: M[x] += f * (x // k)
        # Vectorized:
        M[k:] += f * (np.arange(k, x_max + 1) // k)
    return float(np.max(M[1:]))


def _compute_a(domain, values):
    """Compute a = -sum_k f(k)*log(k)/k."""
    if len(domain) == 0:
        return 0.0
    k = domain.astype(np.float64)
    f = values
    return float(-np.sum(f * np.log(k) / k))


def prime_number_theorem_partial_function():
    """
    Construct a partial function f from a subset of the positive integers to the reals
    that maximizes a hidden number-theoretic objective related to the Prime Number Theorem.

    Returns:
        (domain, values): a tuple of two numpy arrays of equal length describing f.
    """
    t_start = time.time()
    DEADLINE = 600.0  # evaluator kills at 630s; keep safety margin

    def time_left():
        return DEADLINE - (time.time() - t_start)

    # Fallback: the seed's known-feasible construction.
    fallback_domain = np.array([1, 2, 3, 5, 30], dtype=np.int64)
    fallback_values = np.array([3.47155351603266, -1.0, -1.0, -1.0, 1.0], dtype=np.float64)

    best_domain = fallback_domain.copy()
    best_values = fallback_values.copy()
    best_a = _compute_a(best_domain, best_values)

    # Strategy: use the Mobius function mu(k) for k = 1..N. The full Mobius function
    # satisfies sum_{k|x} mu(k) = 0 for x>1, and the identity
    #   sum_{k<=x} mu(k) floor(x/k) = 1  for all x >= 1.
    # For a partial function using mu(k) for k <= N (and 0 for k > N), we have
    #   M(x) = sum_{k<=min(x,N)} mu(k) floor(x/k).
    # For x <= N, M(x) = 1 exactly (feasible). For x > N, M(x) = 1 - sum_{k=N+1}^{x} mu(k) floor(x/k),
    # which can exceed 1. So we must verify feasibility on [1, 10N] and, if violated,
    # fall back to a smaller N or a damped construction.

    # Try increasing N, checking feasibility. Use a deadline guard.
    candidate_Ns = [500, 1000, 2000, 3000, 4000, 5000, 7000, 10000]
    for N in candidate_Ns:
        if time_left() < 60.0:
            break
        mu = _mobius_up_to(N)
        # domain = all k in 1..N with mu(k) != 0 (mu(k)=0 contributes nothing)
        mask = mu[1:] != 0.0
        domain = np.nonzero(mask)[0] + 1
        values = mu[1:][mask]
        K = int(domain.max()) if len(domain) > 0 else 1
        # Check feasibility: max M(x) over x in [1, 10K]
        Mmax = _compute_M_max(domain, values, K)
        if Mmax <= 1.0001:
            a = _compute_a(domain, values)
            if a > best_a:
                best_a = a
                best_domain = domain.copy()
                best_values = values.copy()
        # Even if infeasible, we could try a damped version, but keep it simple: skip.

    # Safety: ensure the returned result is feasible. Re-verify the best candidate.
    if len(best_domain) > 0:
        K = int(best_domain.max())
        Mmax = _compute_M_max(best_domain, best_values, K)
        if Mmax > 1.0001:
            # Fallback to the known-feasible seed.
            best_domain = fallback_domain.copy()
            best_values = fallback_values.copy()

    # Final safety: ensure domain/values are valid (distinct positive ints, K <= 200000).
    if len(best_domain) == 0:
        best_domain = fallback_domain.copy()
        best_values = fallback_values.copy()
    if int(best_domain.max()) > 200000:
        best_domain = fallback_domain.copy()
        best_values = fallback_values.copy()

    return best_domain, best_values


# EVOLVE-BLOCK-END
