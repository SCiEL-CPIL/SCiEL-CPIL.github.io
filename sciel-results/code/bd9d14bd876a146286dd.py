# EVOLVE-BLOCK-START
from math import isqrt, gcd
import time
from functools import lru_cache

def construct():
    N = 256
    deadline = time.time() + 55  # safety margin

    # Sieve to get primes up to N
    is_prime = [True] * (N + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, isqrt(N) + 1):
        if is_prime[i]:
            for j in range(i * i, N + 1, i):
                is_prime[j] = False
    primes = [i for i in range(2, N + 1) if is_prime[i]]

    # For each prime, the largest prime power <= N
    prime_power = {}
    for p in primes:
        power = p
        while power * p <= N:
            power *= p
        prime_power[p] = power

    # Identify which primes can be grouped
    # A prime p can be in a group with q iff p*q <= N
    # For groups of 3: p*q*r <= N
    # For groups of 4: p*q*r*s <= N

    # Small primes: those where p*2 <= N (can potentially pair with 2)
    # But more precisely, we need to identify primes that can be in ANY group of size >= 2
    # A prime p is "groupable" if there exists another prime q such that p*q <= N
    # The smallest other prime is 2, so p is groupable if p*2 <= N, i.e., p <= 128

    # But actually, we need to be more careful. Let's identify primes that can be in groups.
    # For the DP, we'll work with primes that can potentially be grouped.
    # Primes > 128 must be alone (p*2 > 256)

    # Let's identify "small" primes that can be grouped
    # A prime p can be in a group if there's at least one other prime q with p*q <= N
    # Since the smallest prime is 2, p can be in a group iff p <= 128

    # But we should be more precise: which primes can actually benefit from grouping?
    # Let's just include all primes <= 128 in our DP and let the algorithm decide.

    small_primes = [p for p in primes if p <= 128]
    large_primes = [p for p in primes if p > 128]

    # For large primes, they must be alone
    large_sum = sum(prime_power[p] for p in large_primes)
    large_numbers = [prime_power[p] for p in large_primes]

    # For small primes, use bitmask DP to find optimal partition
    # n_small = len(small_primes)
    n_small = len(small_primes)

    # Precompute: for each subset (bitmask), what's the best number we can form?
    # A subset is feasible if the product of its primes <= N
    # The best number for a subset is the max value <= N that uses exactly those primes (each to power >= 1)

    # First, precompute the product for each subset
    subset_product = [0] * (1 << n_small)
    subset_product[0] = 1
    for mask in range(1, 1 << n_small):
        lsb = mask & -mask
        idx = lsb.bit_length() - 1
        p = small_primes[idx]
        subset_product[mask] = subset_product[mask ^ lsb] * p

    # For each feasible subset, compute the best number
    def best_number_for_mask(mask):
        """Find the largest number <= N using exactly the primes in mask, each to power >= 1."""
        if mask == 0:
            return 1
        base = subset_product[mask]
        if base > N:
            return 0  # infeasible

        # Get the primes in this mask
        group_primes = [small_primes[i] for i in range(n_small) if mask & (1 << i)]
        if not group_primes:
            return 1

        # DFS to find the best number
        best = base

        def dfs(current):
            nonlocal best
            if current > best:
                best = current
            if current * min(group_primes) > N:
                return
            for p in group_primes:
                new_val = current * p
                if new_val <= N:
                    dfs(new_val)

        dfs(base)
        return best

    # Precompute best number for all feasible subsets
    subset_best = [0] * (1 << n_small)
    subset_best[0] = 0  # empty set contributes 0
    for mask in range(1, 1 << n_small):
        if subset_product[mask] <= N:
            subset_best[mask] = best_number_for_mask(mask)
        else:
            subset_best[mask] = 0  # infeasible

    # DP: dp[mask] = max sum for the primes represented by mask
    # dp[0] = 0
    # dp[mask] = max over all non-empty submasks sub of mask where sub is feasible:
    #   subset_best[sub] + dp[mask ^ sub]

    dp = [0] * (1 << n_small)
    # For reconstruction, store which submask was chosen
    choice = [0] * (1 << n_small)

    for mask in range(1, 1 << n_small):
        best_val = 0
        best_sub = 0
        # Iterate through all non-empty submasks of mask
        sub = mask
        while sub:
            if subset_best[sub] > 0:  # feasible
                remaining = mask ^ sub
                if dp[remaining] >= 0:
                    val = subset_best[sub] + dp[remaining]
                    if val > best_val:
                        best_val = val
                        best_sub = sub
            sub = (sub - 1) & mask

        dp[mask] = best_val
        choice[mask] = best_sub

    # Reconstruct the solution
    full_mask = (1 << n_small) - 1
    small_numbers = []
    mask = full_mask
    while mask:
        sub = choice[mask]
        if sub == 0:
            # This shouldn't happen if dp is well-formed, but handle gracefully
            break
        small_numbers.append(subset_best[sub])
        mask ^= sub

    # Combine all numbers
    result = [1] + large_numbers + small_numbers

    # Verify: distinct, in range, pairwise coprime
    if len(set(result)) != len(result):
        # Fallback
        return [int(x) for x in [1] + [prime_power[p] for p in primes]]

    if any(x < 1 or x > N for x in result):
        return [int(x) for x in [1] + [prime_power[p] for p in primes]]

    for i in range(len(result)):
        for j in range(i + 1, len(result)):
            if gcd(result[i], result[j]) != 1:
                return [int(x) for x in [1] + [prime_power[p] for p in primes]]

    return [int(x) for x in result]
# EVOLVE-BLOCK-END
