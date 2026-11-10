# EVOLVE-BLOCK-START
import numpy as np
import time


def _compute_mobius(N):
    """Compute Moebius function mu(k) for k = 1..N using linear sieve."""
    mu = np.zeros(N + 1, dtype=np.int8)
    mu[1] = 1
    is_prime = np.ones(N + 1, dtype=bool)
    is_prime[0] = is_prime[1] = False
    primes = []
    for i in range(2, N + 1):
        if is_prime[i]:
            primes.append(i)
            mu[i] = -1
        for p in primes:
            if i * p > N:
                break
            is_prime[i * p] = False
            if i % p == 0:
                mu[i * p] = 0
                break
            else:
                mu[i * p] = -mu[i]
    return mu


def _verify_feasibility(domain, values, max_x):
    """
    Check M(x) = sum_k f(k) * floor(x/k) <= 1.0001 for x in [1, max_x].
    Uses the identity: M(x) = sum_{n=1}^{x} g(n) where g(n) = sum_{k|n} f(k).
    """
    m = len(domain)
    if m == 0:
        return True

    # Compute g(n) for n up to max_x
    g = np.zeros(max_x + 1, dtype=np.float64)
    for i in range(m):
        k = domain[i]
        v = values[i]
        if v != 0.0:
            # Add v to g(n) for all n that are multiples of k
            g[k::k] += v

    # M(x) = cumsum(g)[x]
    M = np.cumsum(g)
    # Check M(x) <= 1.0001 for x in [1, max_x]
    if np.any(M[1:max_x + 1] > 1.0001):
        return False
    return True


def _compute_objective(domain, values):
    """Compute a = -sum_k f(k) * log(k) / k"""
    log_k = np.log(domain.astype(np.float64))
    a = -np.sum(values * log_k / domain.astype(np.float64))
    return a


def prime_number_theorem_partial_function():
    """
    Construct a partial function f from a subset of the positive integers to the reals
    that maximizes a hidden number-theoretic objective related to the Prime Number Theorem.

    Returns:
        (domain, values): a tuple of two numpy arrays of equal length describing f
    """
    start_time = time.time()
    deadline = 600.0

    best_domain = None
    best_values = None
    best_a = -np.inf

    # Try a dense range of K values to find the optimal feasible K
    # Start from small K and go up, trying many values
    # For each K, check feasibility and compute objective

    # Precompute mobius for the maximum K we'll try
    # We'll try K up to 200000
    K_max = 200000
    mu_full = _compute_mobius(K_max)

    # Try K values: use a log-spaced set plus some linear spacing
    # Focus on finding the largest feasible K since objective likely increases with K
    K_candidates = []
    # Small values
    for K in range(10, 200):
        K_candidates.append(K)
    # Medium values - log spaced
    for K in range(200, 5000, 100):
        K_candidates.append(K)
    for K in range(5000, 20000, 500):
        K_candidates.append(K)
    for K in range(20000, 50000, 1000):
        K_candidates.append(K)
    for K in range(50000, 100000, 2000):
        K_candidates.append(K)
    for K in range(100000, 200001, 5000):
        K_candidates.append(K)
    # Ensure 200000 is included
    if K_max not in K_candidates:
        K_candidates.append(K_max)

    # Sort and remove duplicates
    K_candidates = sorted(set(K_candidates))

    for K in K_candidates:
        if time.time() > deadline - 30:
            break

        max_x = 10 * K

        # Get domain and values for this K
        mu_K = mu_full[1:K + 1]
        non_zero = np.where(mu_K != 0)[0] + 1
        if len(non_zero) == 0:
            continue

        domain = non_zero.astype(np.int64)
        values = mu_K[non_zero - 1].astype(np.float64)

        # Verify feasibility
        if not _verify_feasibility(domain, values, max_x):
            continue

        # Compute objective
        a = _compute_objective(domain, values)

        if a > best_a:
            best_a = a
            best_domain = domain.copy()
            best_values = values.copy()

    # Fallback
    if best_domain is None:
        domain = np.array([1, 2, 3, 5, 30], dtype=np.int64)
        values = np.array([3.47155351603266, -1.0, -1.0, -1.0, 1.0], dtype=np.float64)
    else:
        domain = best_domain
        values = best_values

    return domain, values


# EVOLVE-BLOCK-END
