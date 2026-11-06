"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from fractions import Fraction
from itertools import combinations
import random

def _is_valid_exact(S):
    """Check if set S is reciprocal-subsum-free using exact Fraction arithmetic."""
    if len(S) <= 1:
        return True
    S_set = set(S)
    for a in S:
        others = [x for x in S if x != a]
        if not others:
            continue
        target = Fraction(1, a)
        n = len(others)
        # Meet in the middle for larger sets, otherwise brute force
        if n <= 15:
            # Brute force all subsets
            for mask in range(1, 1 << n):
                s = Fraction(0)
                for i in range(n):
                    if mask & (1 << i):
                        s += Fraction(1, others[i])
                if s == target:
                    return False
        else:
            # Meet in the middle
            half = n // 2
            left = others[:half]
            right = others[half:]
            # Compute all subset sums for left
            left_sums = set()
            for mask in range(1 << len(left)):
                s = Fraction(0)
                for i in range(len(left)):
                    if mask & (1 << i):
                        s += Fraction(1, left[i])
                left_sums.add(s)
            # Check if target - right_subset_sum is in left_sums
            for mask in range(1, 1 << len(right)):
                s = Fraction(0)
                for i in range(len(right)):
                    if mask & (1 << i):
                        s += Fraction(1, right[i])
                if (target - s) in left_sums:
                    return False
            # Also check if target itself is in left_sums (empty right subset)
            if target in left_sums:
                return False
    return True

def _is_valid_fast_fp(S):
    """Fast validity check using floating point with early termination (for speed)."""
    S_set = set(S)
    for a in S:
        others = [x for x in S if x != a]
        if not others:
            continue
        target = 1.0 / a
        reciprocals = sorted([1.0 / x for x in others], reverse=True)
        n = len(reciprocals)
        for mask in range(1, 1 << n):
            s = 0.0
            for i in range(n):
                if mask & (1 << i):
                    s += reciprocals[i]
                    if s > target + 1e-12:
                        break
            if abs(s - target) < 1e-10:
                return False
    return True

def _greedy_build(order, deadline):
    """Build a valid set greedily following the given order."""
    S = []
    for x in order:
        if time.time() > deadline:
            break
        candidate = S + [x]
        if _is_valid_fast_fp(candidate):
            S = candidate
    return S

def construct():
    start_time = time.time()
    deadline = 85.0  # safety margin before 90s kill

    best = []

    # Try multiple greedy orderings
    base = list(range(1, 25))

    orders = []
    # Descending
    orders.append(sorted(base, reverse=True))
    # Ascending
    orders.append(sorted(base))
    # Even first then odd
    orders.append([x for x in base if x % 2 == 0] + [x for x in base if x % 2 == 1])
    # Odd first then even
    orders.append([x for x in base if x % 2 == 1] + [x for x in base if x % 2 == 0])
    # Large first
    orders.append(sorted(base, key=lambda x: -x))
    # Mix: alternate large/small
    srt = sorted(base)
    mixed = []
    lo, hi = 0, len(srt) - 1
    while lo <= hi:
        mixed.append(srt[hi])
        hi -= 1
        if lo <= hi:
            mixed.append(srt[lo])
            lo += 1
    orders.append(mixed)
    # Random orderings
    rng = random.Random(42)
    for _ in range(5):
        r = list(base)
        rng.shuffle(r)
        orders.append(r)

    for order in orders:
        if time.time() > deadline:
            break
        S = _greedy_build(order, deadline)
        if len(S) > len(best):
            best = S

    # Local search: try adding elements not in best
    S = list(best)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for x in range(1, 25):
            if x in S:
                continue
            candidate = S + [x]
            if _is_valid_fast_fp(candidate):
                # Verify with exact arithmetic
                if _is_valid_exact(candidate):
                    S = candidate
                    improved = True
                    break

    # Try 1-for-2 swaps
    best2 = list(S)
    for i in range(len(S)):
        if time.time() > deadline:
            break
        for j in range(i + 1, len(S)):
            if time.time() > deadline:
                break
            reduced = [x for k, x in enumerate(S) if k != i and k != j]
            for x in range(1, 25):
                if x in reduced:
                    continue
                for y in range(1, 25):
                    if y in reduced or y == x:
                        continue
                    candidate = reduced + [x, y]
                    if len(candidate) > len(best2) and _is_valid_fast_fp(candidate):
                        if _is_valid_exact(candidate):
                            best2 = candidate

    if len(best2) > len(S):
        S = best2

    # Final polish: try to add more elements
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for x in range(1, 25):
            if x in S:
                continue
            candidate = S + [x]
            if _is_valid_fast_fp(candidate) and _is_valid_exact(candidate):
                S = candidate
                improved = True
                break

    # Verify final answer is valid
    if not _is_valid_exact(S):
        # Fallback to a known valid set
        S = list(range(13, 25))
        if not _is_valid_exact(S):
            S = [13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]

    return S
# EVOLVE-BLOCK-END
