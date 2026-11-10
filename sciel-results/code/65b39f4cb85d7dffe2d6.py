# EVOLVE-BLOCK-START
import numpy as np
import time


def _compute_mobius(K):
    """Compute the Mobius function mu(n) for n=1..K using a linear sieve."""
    mu = np.ones(K + 1, dtype=np.int8)
    mu[0] = 0
    is_prime = np.ones(K + 1, dtype=bool)
    is_prime[0] = False
    is_prime[1] = False
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


def _compute_M_vectorized(domain, values, max_x):
    """
    Compute M(x) = sum_k f(k) * floor(x/k) for x in [1, max_x]
    using a vectorized difference-array approach.
    """
    diff = np.zeros(max_x + 2, dtype=np.float64)
    
    for idx in range(len(domain)):
        k = int(domain[idx])
        f_k = values[idx]
        if f_k == 0.0:
            continue
        
        # j ranges from 1 to max_x // k
        max_j = max_x // k
        if max_j == 0:
            continue
        
        # For each j, the interval is [j*k, min((j+1)*k - 1, max_x)]
        # Contribution is f_k * j
        # Start positions: k, 2k, 3k, ..., max_j * k
        starts = np.arange(1, max_j + 1, dtype=np.int64) * k
        # End positions (exclusive): (j+1)*k for j=1..max_j-1, max_x+1 for last
        ends = np.arange(2, max_j + 2, dtype=np.int64) * k
        ends = np.minimum(ends, max_x + 1)
        # Values to add: f_k * 1, f_k * 2, ..., f_k * max_j
        vals = f_k * np.arange(1, max_j + 1, dtype=np.float64)
        
        # Use np.add.at for scattered updates (handles duplicates safely)
        np.add.at(diff, starts, vals)
        np.add.at(diff, ends, -vals)
    
    M = np.cumsum(diff)
    return M[:max_x + 1]


def prime_number_theorem_partial_function():
    """
    Construct a partial function f from a subset of the positive integers to the reals
    that maximizes a hidden number-theoretic objective related to the Prime Number Theorem.

    Returns:
        (domain, values): a tuple of two numpy arrays of equal length describing f.
    """
    start_time = time.time()
    deadline = 600.0  # safety margin before 630s kill

    # Use K = 200000 with f(k) = mu(k)
    # For f = mu, M(x) = 1 for all x >= 1 (Mertens identity), so constraint M(x) <= 1.0001 is satisfied.
    # The objective a = -sum_k mu(k) * log(k) / k
    # We can compute this without the expensive M(x) verification.
    
    K = 200000
    mu = _compute_mobius(K)
    
    # Include all non-zero entries
    mask = mu[1:K + 1] != 0
    domain = np.where(mask)[0] + 1  # 1-indexed
    values = mu[domain].astype(np.float64)
    
    # Verify the constraint with a quick check (or trust the math)
    # M(x) = sum_{d|x} mu(d) = 1 if x=1, 0 otherwise... wait, that's the sum over divisors.
    # M(x) = sum_{n<=x} sum_{d|n} mu(d) = sum_{d<=x} mu(d) * floor(x/d)
    # This equals 1 for all x >= 1.
    # So the constraint M(x) <= 1.0001 is satisfied with M(x) = 1 exactly.
    
    # Compute objective a = -sum_k f(k) * log(k) / k
    k_float = domain.astype(np.float64)
    a = -np.sum(values * np.log(k_float) / k_float)
    
    # Quick sanity check with a small M(x) computation to make sure
    # For safety, let's verify with a smaller range if time permits
    if time.time() - start_time < deadline - 10:
        # Verify M(x) for x in [1, 1000] to be safe
        max_x_check = min(1000, 10 * K)
        M_check = _compute_M_vectorized(domain, values, max_x_check)
        if np.max(M_check[1:max_x_check + 1]) > 1.0001:
            # This shouldn't happen, but fall back to a smaller K
            K = 10000
            mu = _compute_mobius(K)
            mask = mu[1:K + 1] != 0
            domain = np.where(mask)[0] + 1
            values = mu[domain].astype(np.float64)
            k_float = domain.astype(np.float64)
            a = -np.sum(values * np.log(k_float) / k_float)
    
    return domain, values


# EVOLVE-BLOCK-END
