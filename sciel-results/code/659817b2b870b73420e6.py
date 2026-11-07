import sys
import time
import math
from bisect import bisect_left, bisect_right

# EVOLVE-BLOCK-START
MAX_N = 10000000

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s kill

    # 1. Compute sigma(n) for all n <= MAX_N
    # Pure Python with contiguous stride for cache efficiency
    sigma = [0] * (MAX_N + 1)
    sig = sigma
    limit = MAX_N
    for d in range(1, limit + 1):
        for m in range(d, limit + 1, d):
            sig[m] += d

    if time.time() > deadline:
        return [11, 6]

    # 2. Build dictionary: sigma value -> sorted list of n with that sigma
    # This allows O(log k) lookup of all a with sigma(a) = sigma(b)
    from collections import defaultdict
    sigma_groups = defaultdict(list)
    for n in range(1, limit + 1):
        sigma_groups[sig[n]].append(n)
    # Each list is already sorted since we iterate n in order

    # 3. Search for best pair (a, b)
    # Minimize |a/b - 3/2| = |2a - 3b| / (2b)
    # Let k = |2a - 3b|. Want small k and large b.
    # Iterate b from MAX_N down to 1.

    best_a, best_b = 11, 6
    best_err_num, best_err_den = 4, 12  # Error for 11/6 is 1/3 = 4/12
    best_score_val = 1.0 / 3.0

    # Precompute for speed
    sig_local = sig
    groups = sigma_groups

    for b in range(MAX_N, 0, -1):
        if time.time() > deadline:
            break

        # Pruning: minimum possible error for this b is 1/(2*b)
        # If that's worse than best, we can stop
        if 1.0 / (2.0 * b) >= best_score_val:
            break

        sb = sig_local[b]
        # Get all a with sigma(a) == sb
        L = groups.get(sb)
        if L is None or len(L) == 0:
            continue

        # Target a is 1.5 * b = 3b/2
        # We want a in L closest to 3b/2
        # Binary search for 3b/2 in L
        target = 3 * b / 2.0

        # Find insertion point
        idx = bisect_left(L, target)

        # Check candidates around idx
        # We need k = |2a - 3b| to be small
        # a = 3b/2 ± k/2, so k = 2|a - 3b/2|
        # Check a few positions around idx

        # Positions to check: idx-3 to idx+3 (7 candidates)
        # But we need to ensure a >= 1 and a <= MAX_N
        checked = set()
        for offset in range(-3, 4):
            i = idx + offset
            if i < 0 or i >= len(L):
                continue
            a = L[i]
            if a in checked:
                continue
            checked.add(a)
            if a < 1 or a > MAX_N:
                continue
            k = abs(2 * a - 3 * b)
            if k == 0:
                continue
            # Compare k/(2b) with best_err_num/best_err_den
            # k * best_err_den < best_err_num * 2 * b
            if k * best_err_den < best_err_num * (2 * b):
                best_a, best_b = a, b
                best_err_num, best_err_den = k, 2 * b
                best_score_val = k / (2.0 * b)

        # Early exit: if we found k=1, it's the minimum possible k.
        # For any smaller b', min error is 1/(2b') > 1/(2b) = current best.
        # So we can break.
        if best_err_num == 1:
            break

    return [int(best_a), int(best_b)]

# EVOLVE-BLOCK-END
