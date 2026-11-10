# EVOLVE-BLOCK-START
import numpy as np
import time


def _compute_mobius(n):
    """Compute Mobius function mu(1..n) using sieve."""
    mu = np.ones(n + 1, dtype=np.int8)
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


def prime_number_theorem_partial_function():
    """
    Construct a partial function f from a subset of the positive integers to the reals
    that maximizes a hidden number-theoretic objective related to the Prime Number Theorem.

    Returns:
        (domain, values): a tuple of two numpy arrays of equal length describing f
    """
    start_time = time.time()
    deadline = start_time + 580  # safety margin before 600s limit

    # Strategy: Greedy construction using Mobius-inspired values
    # The optimal solution is the Mobius function mu(k), where:
    #   sum_{k<=x} mu(k) * floor(x/k) = 1 for all x
    #   -sum mu(k) * log(k)/k -> 1 as we include more terms
    #
    # We'll compute mu up to a large bound and use it directly.
    # The constraint M(x) = sum_k f(k)*floor(x/k) <= 1.0001 is satisfied
    # when f = mu because sum_{k|x} mu(k) = 1 for x=1 and 0 otherwise,
    # so M(x) = 1 for all x >= 1.
    #
    # After clipping to [-10,10] and adjusting f(1), we need to verify feasibility.

    # Compute Mobius up to a reasonable bound
    K = 50000
    mu = _compute_mobius(K)

    # Select domain: all k where mu(k) != 0 (square-free numbers)
    mask = mu[1:] != 0
    domain = np.where(mask)[0] + 1  # 1-indexed
    values = mu[1:][mask].astype(np.float64)

    # Clip values to [-10, 10] (they're already -1, 0, 1 so no change)
    values = np.clip(values, -10.0, 10.0)

    # Check deadline
    if time.time() > deadline - 30:
        # Compute S = sum f(k)/k and adjust f(1)
        S = np.sum(values / domain.astype(np.float64))
        if domain[0] == 1:
            idx = np.where(domain == 1)[0][0]
            values[idx] -= S  # f(1) = f(1) - S
        else:
            # If 1 not in domain, f(1) = 0 before adjustment, so f(1) = -S
            # But we can't add 1 to domain if it's not there... 
            # Actually the problem says "f(1) = 0 before this if 1 is not in the domain"
            # So we need to add 1 to domain with value -S
            values = np.concatenate([[ -S ], values])
            domain = np.concatenate([[1], domain])

        # Final feasibility check
        max_K = int(np.max(domain))
        x_max = 10 * max_K
        # M(x) = sum_k f(k) * floor(x/k)
        # Check at a few points
        for x in [1, 2, 3, 5, 10, 100, 1000, max_K, x_max]:
            M_x = np.sum(values * (x // domain))
            if M_x > 1.0001:
                # Infeasible - fall back to simpler solution
                domain = np.array([1, 2, 3, 5, 30], dtype=np.int64)
                values = np.array([3.47155351603266, -1.0, -1.0, -1.0, 1.0], dtype=np.float64)
                return domain, values

        return domain, values

    # Full computation with deadline guard
    # The Mobius function satisfies M(x) = 1 exactly for all x >= 1
    # So it's always feasible. Just need to handle the f(1) adjustment.

    # Compute S = sum f(k)/k over clipped values
    S = np.sum(values / domain.astype(np.float64))

    # Adjust f(1)
    if domain[0] == 1:
        idx = np.where(domain == 1)[0][0]
        values[idx] -= S
    else:
        values = np.concatenate([[ -S ], values])
        domain = np.concatenate([[1], domain])

    # Verify feasibility at key points
    max_K = int(np.max(domain))
    test_points = np.array([1, 2, 3, 4, 5, 10, 50, 100, 500, 1000, 5000, max_K, 10 * max_K])
    for x in test_points:
        M_x = np.sum(values * (x // domain))
        if M_x > 1.0001:
            # Shouldn't happen with Mobius, but fallback
            domain = np.array([1, 2, 3, 5, 30], dtype=np.int64)
            values = np.array([3.47155351603266, -1.0, -1.0, -1.0, 1.0], dtype=np.float64)
            return domain, values

    return domain, values


# EVOLVE-BLOCK-END
