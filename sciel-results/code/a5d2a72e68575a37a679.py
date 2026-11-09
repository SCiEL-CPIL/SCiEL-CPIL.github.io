import time
import numpy as np
from collections import defaultdict
from fractions import Fraction

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 78  # safety margin before 90s

    N = 10_000_000

    # Compute sigma(n) for all n up to N using sieve
    sigma = np.zeros(N + 1, dtype=np.int64)
    for d in range(1, N + 1):
        sigma[d::d] += d
        if d % 1_000_000 == 0:
            if time.time() > deadline - 15:
                break

    # Build groups: dict from sigma value -> list of n
    groups = defaultdict(list)
    for n in range(1, N + 1):
        s = sigma[n]
        groups[s].append(n)

    # Search for pairs minimizing |a/b - 3/2| within each group
    best_error = Fraction(1, 3)  # initial: |11/6 - 3/2| = 1/3
    best_pair = (11, 6)

    target_num = 3
    target_den = 2

    for s_val, members in groups.items():
        if len(members) < 2:
            continue
        if time.time() > deadline - 10:
            break

        members_sorted = sorted(members)

        # For each a in members, find b such that a/b is closest to 3/2
        # i.e., b ≈ 2a/3
        for a in members_sorted:
            # We want a/b ≈ 3/2 => b ≈ 2a/3
            target_b = 2 * a // 3
            # Find candidates near target_b in members_sorted
            # Use binary search
            lo = 0
            hi = len(members_sorted)
            while lo < hi:
                mid = (lo + hi) // 2
                if members_sorted[mid] < target_b:
                    lo = mid + 1
                else:
                    hi = mid
            # Check positions lo-1, lo, lo+1
            for idx in range(max(0, lo - 1), min(len(members_sorted), lo + 2)):
                b = members_sorted[idx]
                if a == b:
                    continue
                # Compute |a/b - 3/2| exactly
                # a/b - 3/2 = (2a - 3b) / (2b)
                num = 2 * a - 3 * b
                den = 2 * b
                if num < 0:
                    num = -num
                err = Fraction(num, den)
                if err < best_error:
                    best_error = err
                    best_pair = (a, b)

    return [int(best_pair[0]), int(best_pair[1])]

# EVOLVE-BLOCK-END
