# EVOLVE-BLOCK-START
import numpy as np
import time


def _compute_mobius(n):
    """Compute Mobius function mu(k) for k=0..n."""
    mu = np.ones(n + 1, dtype=np.int64)
    is_prime = np.ones(n + 1, dtype=bool)
    is_prime[0] = False
    is_prime[1] = False
    primes = []
    for i in range(2, n + 1):
        if is_prime[i]:
            primes.append(i)
        for p in primes:
            if i * p > n:
                break
            is_prime[i * p] = False
            if i % p == 0:
                mu[i * p] = 0
                break
            else:
                mu[i * p] = -mu[i]
    mu[0] = 0
    mu[1] = 1
    return mu


def _check_feasibility(domain, values, K):
    """Check M(x) <= 1.0001 for all x in [1, 10K]."""
    max_x = 10 * K
    m = len(domain)
    M = np.zeros(max_x + 1, dtype=np.float64)
    for i in range(m):
        k = int(domain[i])
        f_k = values[i]
        if f_k == 0.0:
            continue
        max_j = max_x // k
        for j in range(1, max_j + 1):
            start = j * k
            end = min((j + 1) * k, max_x + 1)
            M[start:end] += f_k * j
    return np.max(M[1:]) <= 1.0001


def _evaluate_candidate(domain, values):
    """Compute the objective a after the f(1) adjustment. Returns -inf if infeasible."""
    K = int(domain.max())
    if not _check_feasibility(domain, values, K):
        return -np.inf
    S = np.sum(values / domain.astype(np.float64))
    idx_1 = np.where(domain == 1)[0]
    if len(idx_1) > 0:
        values_adj = values.copy()
        values_adj[idx_1[0]] -= S
    else:
        values_adj = values.copy()
    a = -np.sum(values_adj * np.log(domain.astype(np.float64)) / domain.astype(np.float64))
    return a


def prime_number_theorem_partial_function():
    """
    Construct a partial function f from a subset of the positive integers to the reals
    that maximizes a hidden number-theoretic objective related to the Prime Number Theorem.
    """
    start_time = time.time()
    deadline = start_time + 570  # 30s safety margin

    best_score = -np.inf
    best_domain = None
    best_values = None

    # Seed solution
    domain_seed = np.array([1, 2, 3, 5, 30], dtype=np.int64)
    values_seed = np.array([3.47155351603266, -1.0, -1.0, -1.0, 1.0], dtype=np.float64)
    score = _evaluate_candidate(domain_seed, values_seed)
    if score > best_score:
        best_score = score
        best_domain = domain_seed.copy()
        best_values = values_seed.copy()

    # Try Mobius-based solutions for increasing N
    for N in [10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000, 100000, 200000]:
        if time.time() > deadline:
            break

        mu = _compute_mobius(N)
        mask = mu[1:] != 0
        if not np.any(mask):
            continue
        domain = np.where(mask)[0] + 1
        domain = domain.astype(np.int64)
        values = mu[1:][mask].astype(np.float64)
        K = int(domain.max())

        score = _evaluate_candidate(domain, values)
        if score > best_score:
            best_score = score
            best_domain = domain.copy()
            best_values = values.copy()

    if best_domain is None:
        return domain_seed, values_seed

    return best_domain, best_values


# EVOLVE-BLOCK-END
