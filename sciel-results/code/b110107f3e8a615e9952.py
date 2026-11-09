"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    import time
    from fractions import Fraction
    from math import lcm as _lcm

    deadline = time.time() + 75.0  # safety margin before 90s wall-clock kill

    denominators = list(range(2, 49))  # [2, 3, ..., 48]
    n = len(denominators)  # 47

    # Compute LCM of all denominators to convert to integer weights
    L = 1
    for d in denominators:
        L = _lcm(L, d)

    weights = [L // d for d in denominators]  # weight[i] = L / (i+2)
    target = L  # sum must be < target

    # Meet-in-the-middle: split into two halves
    mid = n // 2  # 23
    left = weights[:mid]
    right = weights[mid:]

    # Generate all subset sums for left half
    left_sums = []  # list of (sum, bitmask)
    for mask in range(1 << mid):
        s = 0
        m = mask
        i = 0
        while m:
            if m & 1:
                s += left[i]
            m >>= 1
            i += 1
        left_sums.append((s, mask))

    # Sort left sums by value
    left_sums.sort(key=lambda x: x[0])

    # For each right subset, find the best left subset such that left_sum + right_sum < target
    best_deficit = None  # target - (left_sum + right_sum), we want to minimize this
    best_left_mask = 0
    best_right_mask = 0

    for mask_r in range(1 << (n - mid)):
        s_r = 0
        m = mask_r
        i = 0
        while m:
            if m & 1:
                s_r += right[i]
            m >>= 1
            i += 1

        if s_r >= target:
            continue

        # We need left_sum < target - s_r
        # Find the largest left_sum < target - s_r
        limit = target - s_r  # we need left_sum < limit, i.e., left_sum <= limit - 1

        # Binary search for the rightmost left_sum < limit
        lo, hi = 0, len(left_sums) - 1
        ans = -1
        while lo <= hi:
            mid_i = (lo + hi) // 2
            if left_sums[mid_i][0] < limit:
                ans = mid_i
                lo = mid_i + 1
            else:
                hi = mid_i - 1

        if ans >= 0:
            s_l, mask_l = left_sums[ans]
            total = s_l + s_r
            if total == 0:
                continue
            deficit = target - total
            if best_deficit is None or deficit < best_deficit:
                best_deficit = deficit
                best_left_mask = mask_l
                best_right_mask = mask_r

    if best_deficit is None:
        # Fallback: greedy
        selected = []
        total = Fraction()
        for i, d in enumerate(denominators):
            if total + Fraction(1, d) < 1:
                selected.append(i)
                total += Fraction(1, d)
        return selected

    # Build index list
    indices = []
    for i in range(mid):
        if best_left_mask & (1 << i):
            indices.append(i)
    for i in range(n - mid):
        if best_right_mask & (1 << i):
            indices.append(mid + i)

    if not indices:
        # Fallback
        return [0]

    return indices
# EVOLVE-BLOCK-END
