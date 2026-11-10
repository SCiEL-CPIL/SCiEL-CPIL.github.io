# EVOLVE-BLOCK-START
import numpy as np
import time


def _compute_mobius(K):
    """Compute the Mobius function mu(n) for n=1..K using a linear sieve."""
    mu = np.ones(K + 1, dtype=np.int64)
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


def _compute_M_fast(domain, values, max_x):
    """
    Compute M(x) = sum_k f(k) * floor(x/k) for x in [1, max_x]
    using an optimized difference-array approach.
    
    Key insight: for each k, the net contribution to the diff array is
    simply adding f(k) to positions k, 2k, 3k, ..., floor(max_x/k)*k.
    This is because the +f(k)*j from interval j and -f(k)*(j-1) from
    interval j-1 cancel to give net +f(k) at position j*k.
    """
    diff = np.zeros(max_x + 2, dtype=np.float64)
    
    for idx in range(len(domain)):
        k = int(domain[idx])
        f_k = values[idx]
        if f_k == 0.0:
            continue
        max_j = max_x // k
        if max_j == 0:
            continue
        # Add f_k to positions k, 2k, ..., max_j*k
        diff[k : (max_j + 1) * k : k] += f_k
    
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

    # Try increasing K until we run out of time
    best_result = None
    best_a = -np.inf

    # Start with a reasonable K and increase
    for K in [5000, 10000, 20000, 50000, 100000, 200000]:
        if time.time() - start_time > deadline - 90:
            break

        # Compute Mobius function
        mu = _compute_mobius(K)

        # Only include non-zero entries to keep m small
        mask = mu[1:K + 1] != 0
        domain = np.where(mask)[0] + 1  # 1-indexed
        values = mu[domain].astype(np.float64)

        if len(domain) == 0:
            continue

        # Simulate evaluator: clip to [-10, 10], adjust f(1)
        clipped = np.clip(values, -10, 10)
        k_float = domain.astype(np.float64)
        S = np.sum(clipped / k_float)
        
        if domain[0] == 1:
            adjusted = clipped.copy()
            adjusted[0] -= S
            adj_domain = domain
        else:
            adjusted = np.concatenate([[-S], clipped])
            adj_domain = np.concatenate([[1], domain])
        
        # Compute M(x) for x in [1, 10K]
        max_x = 10 * K
        M = _compute_M_fast(adj_domain, adjusted, max_x)

        # Check constraint: M(x) <= 1.0001 for all x in [1, 10K]
        if np.max(M[1:max_x + 1]) > 1.0001:
            continue

        # Compute objective a = -sum_k f(k) * log(k) / k using adjusted values
        adj_k = adj_domain.astype(np.float64)
        a = -np.sum(adjusted * np.log(adj_k) / adj_k)

        if a > best_a:
            best_a = a
            best_result = (domain, values)

        if time.time() - start_time > deadline - 60:
            break

    if best_result is None:
        # Fallback to the seed
        domain = np.array([1, 2, 3, 5, 30], dtype=np.int64)
        values = np.array([3.47155351603266, -1.0, -1.0, -1.0, 1.0], dtype=np.float64)
        return domain, values

    return best_result


# EVOLVE-BLOCK-END
