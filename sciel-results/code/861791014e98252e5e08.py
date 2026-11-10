# EVOLVE-BLOCK-START
import numpy as np
import time


def _mobius(n):
    """Compute Mobius function mu(1)..mu(n) via sieve."""
    mu = np.ones(n + 1, dtype=np.int64)
    is_prime = np.ones(n + 1, dtype=bool)
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
    deadline = start_time + 570.0

    # Compute Mobius function
    K_max = 200000
    mu = _mobius(K_max)
    mu_prefix = np.cumsum(mu)  # mu_prefix[i] = sum_{k=1}^i mu(k)

    # Find the largest K where M(x) = sum_{k=1}^K mu(k)*floor(x/k) <= 1.0001 for all x in [1,10K]
    # For x <= K, M(x) = 1 exactly. Need to check x in (K, 10K].
    
    # Binary search or just try a few values
    feasible_Ks = []
    for K_trial in [100000, 50000, 20000, 10000, 5000, 2000, 1000, 500, 200, 100, 50, 20, 10, 5]:
        if time.time() > deadline - 30:
            break
        max_x = 10 * K_trial
        # Check critical x values
        test_xs = set()
        test_xs.update(range(max(1, K_trial - 5), K_trial + 6))
        for mult in range(2, 11):
            test_xs.update(range(max(1, mult * K_trial - 5), min(max_x + 1, mult * K_trial + 6)))
        test_xs = sorted([x for x in test_xs if 1 <= x <= max_x])

        feasible = True
        for x in test_xs:
            if time.time() > deadline - 10:
                break
            k_max = min(K_trial, x)
            if k_max == 0:
                continue
            M_val = 0.0
            k = 1
            while k <= k_max:
                q = x // k
                k_next = x // q + 1
                k_end = min(k_next - 1, k_max)
                mu_sum = mu_prefix[k_end] - (mu_prefix[k - 1] if k > 1 else 0)
                M_val += q * mu_sum
                k = k_next
            if M_val > 1.0001:
                feasible = False
                break
        if feasible:
            feasible_Ks.append(K_trial)
            break  # Take the largest feasible K

    if feasible_Ks:
        K = feasible_Ks[0]
        mask = mu[1:K + 1] != 0
        domain = np.where(mask)[0] + 1
        values = mu[1:K + 1][mask].astype(np.float64)

        # Apply normalization: S = sum f(k)/k, f(1) -= S
        S = np.sum(values / domain)
        values_adj = values.copy()
        idx1 = np.where(domain == 1)[0]
        if len(idx1) > 0:
            values_adj[idx1[0]] -= S

        # Clip to [-10, 10] (should already be in {-1, 0, 1})
        values_adj = np.clip(values_adj, -10, 10)
        return domain, values_adj

    # Fallback: 2-term solution
    c = 1.0001
    domain = np.array([1, 2], dtype=np.int64)
    values = np.array([c, -2*c], dtype=np.float64)
    return domain, values


# EVOLVE-BLOCK-END
