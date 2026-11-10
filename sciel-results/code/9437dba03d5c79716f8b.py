# EVOLVE-BLOCK-START
import numpy as np
import time


def prime_number_theorem_partial_function():
    """
    Construct a partial function f that maximizes the PNT-related objective.
    Uses the Möbius function truncated at K=200000, choosing the K that
    maximizes the partial sum of mu(k)/k to get the best possible score.
    """
    start_time = time.time()
    deadline = start_time + 600.0

    K = 200000

    # Sieve for Möbius function
    mu = np.ones(K + 1, dtype=np.int32)
    is_prime = np.ones(K + 1, dtype=bool)
    is_prime[0] = False
    is_prime[1] = False
    primes = []

    for i in range(2, K + 1):
        if is_prime[i]:
            primes.append(i)
            for j in range(i, K + 1, i):
                is_prime[j] = False
                mu[j] *= -1
            i2 = i * i
            if i2 <= K:
                for j in range(i2, K + 1, i2):
                    mu[j] = 0

    # Compute A(k) = sum_{d=1}^{k} mu(d)/d
    mu_float = mu.astype(np.float64)
    inv_k = 1.0 / np.arange(1, K + 1, dtype=np.float64)
    terms = mu_float * inv_k
    A = np.cumsum(terms)

    # Find the K_best that maximizes A(k)
    K_best = int(np.argmax(A)) + 1

    elapsed = time.time() - start_time
    if elapsed > deadline - 5.0:
        # Return early with a simple feasible baseline
        domain = np.array([1, 2, 3, 5, 30], dtype=np.int64)
        values = np.array([3.47155351603266, -1.0, -1.0, -1.0, 1.0], dtype=np.float64)
        return domain, values

    # Feasibility check: compute M(x) = sum_{k<=K_best} mu(k) * floor(x/k) for x in [1, 10*K_best]
    # We only need to check that max M(x) <= 1.0001
    # M(x) for x <= K: sum_{k<=x} mu(k)*floor(x/k) = 1 (classic identity)
    # M(x) for x > K: need to verify. Use the divisor-sum approach for a subset.
    # For efficiency, check x values in [K_best+1, 10*K_best] at a coarser grid first.

    Kb = K_best
    max_x = 10 * Kb
    mu_kb = mu[:Kb]  # mu[0..Kb-1], so indices 0..Kb-1 correspond to k=1..Kb

    # Check at x = Kb, 2Kb, 3Kb, ..., 10Kb and some intermediate points
    # Actually, the maximum of M(x) for x > K is likely near x = K or just after.
    # Let's check x in a reasonable set.
    # M(x) = sum_{k=1}^{K} mu(k) * floor(x/k)
    # For a fixed x, this is O(K). Checking 10*K values would be O(K * 10*K) = O(10*K^2) which is too slow.
    # Instead, we check a sparse set of x values.

    # The key insight: M(x) = 1 + sum_{k=1}^{K} mu(k) * (floor(x/k) - 1) for x >= K
    # = 1 + sum_{k=1}^{K} mu(k) * floor((x-k)/k) ... not simpler.
    
    # Alternative: M(x) for x > K can be written as:
    # M(x) = sum_{k=1}^{K} mu(k) * floor(x/k)
    # Let's just sample x values.
    
    sample_xs = np.concatenate([
        np.arange(Kb, Kb + 100, 1),
        np.arange(Kb + 100, max_x + 1, max(1, (max_x - Kb) // 500))
    ])
    sample_xs = np.unique(sample_xs[sample_xs >= 1])

    feasible = True
    for x in sample_xs:
        # M(x) = sum_{k=1}^{K} mu(k) * floor(x/k)
        # Vectorized
        k_arr = np.arange(1, Kb + 1)
        floors = x // k_arr
        Mx = np.dot(mu_kb, floors)
        if Mx > 1.0001:
            feasible = False
            break

    if not feasible:
        # Try smaller K values
        for K_try in range(K_best - 1, max(1, K_best - 10000), -1):
            mu_try = mu[:K_try]
            max_x_try = 10 * K_try
            sample_xs_try = np.concatenate([
                np.arange(K_try, K_try + 50, 1),
                np.arange(K_try + 50, max_x_try + 1, max(1, (max_x_try - K_try) // 200))
            ])
            sample_xs_try = np.unique(sample_xs_try[sample_xs_try >= 1])
            
            ok = True
            for x in sample_xs_try:
                k_arr = np.arange(1, K_try + 1)
                floors = x // k_arr
                Mx = np.dot(mu_try, floors)
                if Mx > 1.0001:
                    ok = False
                    break
            if ok:
                K_best = K_try
                break

    domain = np.arange(1, K_best + 1, dtype=np.int64)
    values = mu[:K_best].astype(np.float64)

    return domain, values


# EVOLVE-BLOCK-END
