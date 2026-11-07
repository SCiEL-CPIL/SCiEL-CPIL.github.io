import numpy as np
import time
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin before 60s kill

    N = 10**7

    # Generate candidate pool: all squares and twice-squares up to N
    # squares: k^2 <= N => k <= floor(sqrt(N))
    # twice-squares: 2*k^2 <= N => k <= floor(sqrt(N/2))
    max_k_sq = int(N**0.5)
    max_k_2sq = int((N // 2)**0.5)

    # Build the candidate set
    candidates = set()
    for k in range(1, max_k_sq + 1):
        candidates.add(k * k)
    for k in range(1, max_k_2sq + 1):
        candidates.add(2 * k * k)

    cand_list = sorted(candidates)
    n_cand = len(cand_list)

    # Compute sigma(n) for each candidate using trial division (fast since candidates are structured)
    # For a square k^2: sigma(k^2) = product over primes p^a || k of (p^(2a+1)-1)/(p-1)
    # For 2*k^2: sigma(2*k^2) = 3 * sigma(k^2) if k is odd, or more generally compute via factorization

    # Actually, let's just compute sigma via a fast method for each candidate
    # Since candidates are ~47k, and each is <= 10^7, trial division up to sqrt(n) is fine

    def sigma_fast(n):
        if n == 0:
            return 0
        result = 1
        temp = n
        # Factor out 2
        if temp % 2 == 0:
            power = 1
            while temp % 2 == 0:
                temp //= 2
                power *= 2
            result *= (power * 2 - 1)  # sum of 1+2+4+...+2^a = 2^(a+1)-1
        # Factor out odd primes
        f = 3
        while f * f <= temp:
            if temp % f == 0:
                power = 1
                while temp % f == 0:
                    temp //= f
                    power *= f
                result *= (power * f - 1) // (f - 1)
            f += 2
        if temp > 1:
            result *= (1 + temp)
        return result

    if time.time() > deadline:
        return [1, 1]

    # Compute sigma for all candidates
    sigma_map = {}
    for i, n in enumerate(cand_list):
        if time.time() > deadline - 10.0:
            break
        s = sigma_fast(n)
        sigma_map[n] = s

    if time.time() > deadline:
        return [1, 1]

    # Group candidates by sigma value
    sigma_groups = defaultdict(list)
    for n, s in sigma_map.items():
        sigma_groups[s].append(n)

    # For each group with >= 2 elements, find the pair minimizing |a/b - 3/2|
    best_err = float('inf')
    best_a = 1
    best_b = 1

    for s, group in sigma_groups.items():
        if time.time() > deadline - 5.0:
            break
        k = len(group)
        if k < 2:
            continue

        group_sorted = sorted(group)
        # For each b in group, find a in group that minimizes |a/b - 3/2|
        # i.e., a closest to 1.5*b
        # Use binary search in the sorted group
        for b in group_sorted:
            target = 1.5 * b
            # Find insertion point
            lo = 0
            hi = k
            while lo < hi:
                mid = (lo + hi) // 2
                if group_sorted[mid] < target:
                    lo = mid + 1
                else:
                    hi = mid
            # Check lo and lo-1
            for idx in [lo - 1, lo]:
                if 0 <= idx < k:
                    a = group_sorted[idx]
                    if a == b:
                        continue
                    err = abs(2 * a - 3 * b) / (2 * b)
                    if err < best_err:
                        best_err = err
                        best_a = a
                        best_b = b

    if best_a == best_b or best_err == float('inf'):
        # Fallback: find any valid pair from the groups
        for s, group in sigma_groups.items():
            if len(group) >= 2:
                best_a = group[0]
                best_b = group[1]
                break
        if best_a == best_b:
            return [1, 1]

    return [int(best_a), int(best_b)]
