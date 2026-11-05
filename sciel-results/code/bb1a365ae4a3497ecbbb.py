"""Find largest dissociated reciprocal set from [1,22] via fast integer backtracking."""
import time
from math import gcd
from functools import reduce

def _lcm(a, b):
    return a // gcd(a, b) * b

# Precompute LCM of 1..22
_LCM = 1
for _i in range(1, 23):
    _LCM = _lcm(_LCM, _i)

# a[i] = LCM / i  (integer weights for 1/i)
_A = [_LCM // i for i in range(1, 23)]

# Known-good fallback (size 9)
_FALLBACK = [1, 2, 3, 5, 7, 11, 13, 17, 19]


def construct():
    deadline = time.time() + 80.0  # safety margin before 90s wall-clock
    best = list(_FALLBACK)
    best_size = len(best)

    # Order candidates: larger reciprocals (smaller i) first tends to build
    # distinct sums faster and prunes sooner.
    order = list(range(22))  # index 0 -> i=1, index 21 -> i=22

    def backtrack(idx, chosen, sums, size):
        nonlocal best, best_size, deadline
        if time.time() > deadline:
            return
        # Prune: even if we take every remaining element we can't beat best
        if size + (22 - idx) <= best_size:
            return
        if idx == 22:
            if size > best_size:
                best_size = size
                best = list(chosen)
            return

        # Option 1: skip element idx
        backtrack(idx + 1, chosen, sums, size)

        # Option 2: take element idx
        w = _A[idx]
        # New subset sums = old sums union (old sums + w)
        new_sums = set(sums)
        conflict = False
        # Fast conflict check: if any (s + w) is already in sums, collision
        for s in sums:
            if (s + w) in sums:
                conflict = True
                break
        if not conflict:
            # Build the expanded set
            for s in sums:
                ns = s + w
                if ns in new_sums:
                    # shouldn't happen given the check, but be safe
                    conflict = True
                    break
                new_sums.add(ns)
        if not conflict:
            chosen.append(idx)
            backtrack(idx + 1, chosen, new_sums, size + 1)
            chosen.pop()

    backtrack(0, [], {0}, 0)

    # Convert indices back to values (index i -> value i+1)
    result = [i + 1 for i in best]
    return result if result else _FALLBACK
