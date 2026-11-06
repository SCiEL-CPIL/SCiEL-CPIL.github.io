"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    import time
    import math
    import bisect
    import sys
    from fractions import Fraction

    start_time = time.time()
    deadline = start_time + 55  # 55s deadline for safety (60s limit)

    denominators = list(range(2, 49))  # [2, 3, ..., 48]
    n = len(denominators)

    # Calculate LCM of 2..48 for exact integer arithmetic
    L = 1
    for d in denominators:
        L = L * d // math.gcd(L, d)

    # Weights: w[i] = L / denominators[i], so sum of selected weights / L = sum of reciprocals
    weights = [L // d for d in denominators]

    # Split into two halves for meet-in-the-middle
    mid = n // 2  # 23
    n1 = mid
    n2 = n - mid  # 24

    weights1 = weights[:n1]
    weights2 = weights[n1:]

    # Generate all subset sums for first half with masks
    # Pack as (sum << n1) | mask to allow efficient sorting and binary search
    sums1 = [0]
    for i in range(n1):
        w = weights1[i]
        current_len = len(sums1)
        for j in range(current_len):
            val = sums1[j]
            sums1.append(val + (w << n1) + (1 << i))

    sums1.sort()

    # Generate all subset sums for second half
    sums2 = [0]
    for i in range(n2):
        w = weights2[i]
        current_len = len(sums2)
        for j in range(current_len):
            val = sums2[j]
            sums2.append(val + (w << n2) + (1 << i))

    mask1_full = (1 << n1) - 1
    mask2_full = (1 << n2) - 1

    best_sum = 0
    best_mask1 = 0
    best_mask2 = 0

    # For each subset sum of second half, find best compatible sum from first half
    # We want s1 + s2 < L, maximizing s1 + s2
    for val2 in sums2:
        if time.time() > deadline:
            break
        s2 = val2 >> n2
        if s2 >= L:
            continue
        m2 = val2 & mask2_full

        limit = L - s2  # we need s1 < limit
        key = limit << n1
        pos = bisect.bisect_left(sums1, key)
        if pos > 0:
            val1 = sums1[pos - 1]
            s1 = val1 >> n1
            m1 = val1 & mask1_full

            total_sum = s1 + s2
            if total_sum > best_sum:
                best_sum = total_sum
                best_mask1 = m1
                best_mask2 = m2

    # Reconstruct indices
    indices = []
    for i in range(n1):
        if best_mask1 & (1 << i):
            indices.append(i)

    for i in range(n2):
        if best_mask2 & (1 << i):
            indices.append(n1 + i)

    # Ensure non-empty
    if not indices:
        indices = [0]

    # Verify with exact arithmetic that sum is strictly below 1
    unit_fracs = [Fraction(1, d) for d in denominators]
    total = sum(unit_fracs[i] for i in indices)
    if total >= 1:
        # This shouldn't happen, but fallback
        # Remove the largest element to get strictly below 1
        indices.sort(key=lambda i: -unit_fracs[i])
        indices = indices[:-1]
        if not indices:
            indices = [0]

    return sorted(indices)
# EVOLVE-BLOCK-END
