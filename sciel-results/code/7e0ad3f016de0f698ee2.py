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
    Accounts for the evaluator's adjustment: f(1) is replaced by f(1) - S
    where S = sum_k f(k)/k (over clipped values).
    If 1 is not in domain, it is added with value -S.
    """
    m = len(domain)
    if m == 0:
        return True

    # Clip values to [-10, 10] as the evaluator does
    clipped_values = np.clip(values, -10.0, 10.0)

    # Compute S = sum_k f(k)/k over clipped values
    S = np.sum(clipped_values / domain.astype(np.float64))

    # Build the adjusted domain and values
    has_one = np.any(domain == 1)
    
    if has_one:
        idx_one = np.where(domain == 1)[0][0]
        adj_domain = domain.copy()
        adj_values = clipped_values.copy()
        adj_values[idx_one] = clipped_values[idx_one] - S
    else:
        # Add 1 to domain with value -S
        adj_domain = np.concatenate([[1], domain])
        adj_values = np.concatenate([[-S], clipped_values])

    # Compute M(x) = sum_k f(k) * floor(x/k) using g(n) approach
    g = np.zeros(max_x + 1, dtype=np.float64)
    for i in range(len(adj_domain)):
        k = adj_domain[i]
        v = adj_values[i]
        if v != 0.0:
            g[k::k] += v

    M = np.cumsum(g)
    if np.any(M[1:max_x + 1] > 1.0001):
        return False
    return True


def _compute_objective(domain, values):
    """Compute a = -sum_k f(k) * log(k) / k with evaluator's adjustment."""
    # Clip values
    clipped_values = np.clip(values, -10.0, 10.0)
    
    # Compute S
    S = np.sum(clipped_values / domain.astype(np.float64))
    
    # Build adjusted domain and values
    has_one = np.any(domain == 1)
    
    if has_one:
        idx_one = np.where(domain == 1)[0][0]
        adj_domain = domain.copy()
        adj_values = clipped_values.copy()
        adj_values[idx_one] = clipped_values[idx_one] - S
    else:
        adj_domain = np.concatenate([[1], domain])
        adj_values = np.concatenate([[-S], clipped_values])
    
    # Compute a = -sum_k f(k) * log(k) / k
    log_k = np.log(adj_domain.astype(np.float64))
    a = -np.sum(adj_values * log_k / adj_domain.astype(np.float64))
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

    # Try K values: focus on finding feasible K values
    # Start with a broad search, then refine
    K_candidates = []
    # Small values - dense
    for K in range(2, 100):
        K_candidates.append(K)
    # Medium values
    for K in range(100, 1000, 10):
        K_candidates.append(K)
    for K in range(1000, 5000, 50):
        K_candidates.append(K)
    for K in range(5000, 20000, 200):
        K_candidates.append(K)
    for K in range(20000, 100000, 1000):
        K_candidates.append(K)
    for K in range(100000, 200001, 5000):
        K_candidates.append(K)
    # Ensure 200000 is included
    if K_max not in K_candidates:
        K_candidates.append(K_max)

    # Sort and remove duplicates
    K_candidates = sorted(set(K_candidates))

    for K in K_candidates:
        if time.time() > deadline - 60:
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
