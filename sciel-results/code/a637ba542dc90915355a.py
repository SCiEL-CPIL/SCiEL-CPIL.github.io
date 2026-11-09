"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
from fractions import Fraction
from collections import defaultdict

P = {'maximum': 10000000, 'target': [3, 2]}

def construct():
    start = time.time()
    DEADLINE = start + 85.0  # safety margin

    MAXN = P['maximum']
    target_num, target_den = P['target']  # 3, 2

    # 1. Compute sigma(n) for all n <= MAXN using a sieve
    sigma = [0] * (MAXN + 1)
    for d in range(1, MAXN + 1):
        for m in range(d, MAXN + 1, d):
            sigma[m] += d

    # 2. Group indices by sigma value
    groups = defaultdict(list)
    for i in range(1, MAXN + 1):
        groups[sigma[i]].append(i)

    # 3. Search for pairs (a, b) with sigma(a) == sigma(b) minimizing |a/b - 3/2|
    #    |a/b - 3/2| = |2a - 3b| / (2b)
    #    We want to minimize this.

    best_err = Fraction(1, 1)  # start with a large error
    best_pair = (11, 6)  # fallback (sigma(11)=12, sigma(6)=12)

    # Strategy: For each group with multiple elements, try to find pairs
    # that minimize |2a - 3b| / (2b).
    #
    # A smart approach: for each b in a group, the ideal a is 3b/2.
    # So for each b, look for a in the same group closest to 3b/2.
    #
    # To make this efficient, sort each group and use binary search.

    # Pre-sort all groups
    for key in groups:
        groups[key].sort()

    # Process groups in order of size (smaller groups first for early results)
    # Actually, let's just iterate through all groups
    group_keys = sorted(groups.keys(), key=lambda k: len(groups[k]))

    for key in group_keys:
        if time.time() > DEADLINE:
            break
        lst = groups[key]
        if len(lst) < 2:
            continue

        # For each b in lst, find a in lst closest to 3b/2
        # Since lst is sorted, we can use binary search
        for b in lst:
            if time.time() > DEADLINE:
                break
            ideal_a = 3 * b / 2.0
            # Find insertion point
            # Use bisect
            lo, hi = 0, len(lst)
            while lo < hi:
                mid = (lo + hi) // 2
                if lst[mid] < ideal_a:
                    lo = mid + 1
                else:
                    hi = mid
            # lo is the first index where lst[lo] >= ideal_a
            candidates = []
            if lo < len(lst):
                candidates.append(lst[lo])
            if lo > 0:
                candidates.append(lst[lo - 1])

            for a in candidates:
                if a == b:
                    continue
                # Check sigma(a) == sigma(b) -- guaranteed by group
                # Compute error
                # |a/b - 3/2| = |2a - 3b| / (2b)
                num = abs(2 * a - 3 * b)
                den = 2 * b
                if num == 0:
                    # a/b = 3/2 exactly -- invalid per task
                    continue
                err = Fraction(num, den)
                if err < best_err:
                    best_err = err
                    best_pair = (a, b)

    # Also try swapping roles: for each a, find b closest to 2a/3
    # This is the same search but from the other direction
    # Actually the above already covers both since we iterate all b
    # and find a closest to 3b/2. But let's also check the reverse
    # for completeness (in case a > b gives better ratio)

    # Actually, the above search considers all (a,b) pairs where a is near 3b/2.
    # This covers both a > b and a < b cases.

    # Final validation
    a, b = best_pair
    if not isinstance(a, int) or not isinstance(b, int):
        a, b = 11, 6
    if a <= 0 or b <= 0 or a > MAXN or b > MAXN:
        a, b = 11, 6
    if sigma[a] != sigma[b]:
        a, b = 11, 6
    if a == b:
        a, b = 11, 6

    return [int(a), int(b)]
# EVOLVE-BLOCK-END
