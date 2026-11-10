# EVOLVE-BLOCK-START
import numpy as np
import time


def _mobius_sieve(n):
    """Compute Mobius function for 1..n using linear sieve."""
    mu = np.ones(n + 1, dtype=np.int64)
    mu[0] = 0
    is_prime = np.ones(n + 1, dtype=bool)
    is_prime[:2] = False
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


def prime_number_theorem_partial_function():
    """
    Construct a partial function f from a subset of the positive integers to the reals
    that maximizes a hidden number-theoretic objective related to the Prime Number Theorem.

    Uses the full Möbius function up to K=200000 to maximize the objective
    a = -sum_k f(k)*log(k)/k while maintaining M(x) <= 1.0001.
    """
    start_time = time.time()
    deadline = start_time + 550  # Safety margin before 600s limit

    # Use maximum K to maximize the objective
    K = 200000
    mu = _mobius_sieve(K)

    # Include all k where mu(k) != 0 (square-free numbers)
    mask = mu != 0
    domain = np.where(mask)[0].astype(np.int64)
    values = mu[domain].astype(np.float64)

    # Ensure 1 is in domain (it should be since mu(1)=1)
    if len(domain) == 0 or domain[0] != 1:
        domain = np.concatenate([[1], domain])
        values = np.concatenate([[1.0], values])

    K_max = int(domain[-1])
    x_max_eval = 10 * K_max  # Full evaluation range: [1, 2000000]

    # Verify feasibility: check M(x) at strategic points
    # For the exact Möbius function, M(x) = 1 for all x >= 1
    # But we need to verify numerically due to floating point
    
    # Check at key points: small x, around K, and at 10K
    check_points = [1, 2, 3, 5, 10, 50, 100, 500, 1000, 5000, 10000,
                    50000, 100000, 200000, 500000, 1000000, 1500000, 2000000]
    check_points = [x for x in check_points if 1 <= x <= x_max_eval]
    
    domain_float = domain.astype(np.float64)
    
    feasible = True
    for x in check_points:
        # M(x) = sum_k f(k) * floor(x/k)
        floors = np.floor(x / domain_float).astype(np.int64)
        Mx = np.sum(values * floors)
        if Mx > 1.0001:
            feasible = False
            break
    
    if not feasible:
        # Fallback to a small known-feasible set
        domain = np.array([1, 2, 3, 5, 30], dtype=np.int64)
        values = np.array([3.47155351603266, -1.0, -1.0, -1.0, 1.0], dtype=np.float64)
        return domain, values

    # Check time - if we're running low, return immediately
    elapsed = time.time() - start_time
    if elapsed > deadline - 10:
        return domain, values

    # The objective a = -sum_k f(k)*log(k)/k is unchanged by f(1) adjustment
    # since log(1) = 0
    log_k = np.log(domain_float)
    a = -np.sum(values * log_k / domain_float)

    return domain, values


# EVOLVE-BLOCK-END
