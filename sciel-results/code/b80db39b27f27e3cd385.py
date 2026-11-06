"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from itertools import combinations

def _is_valid(S):
    """Check if set S is reciprocal-subsum-free."""
    S_set = set(S)
    for a in S:
        others = [x for x in S if x != a]
        if not others:
            continue
        target = 1.0 / a
        # Check all non-empty subsets of others
        n = len(others)
        # Use subset-sum style: enumerate all subsets
        for mask in range(1, 1 << n):
            s = 0.0
            for i in range(n):
                if mask & (1 << i):
                    s += 1.0 / others[i]
                    if s > target + 1e-12:
                        break
            if abs(s - target) < 1e-10:
                return False
    return True

def _is_valid_fast(S):
    """Faster validity check using sorted reciprocals and early termination."""
    S_set = set(S)
    for a in S:
        others = [x for x in S if x != a]
        if not others:
            continue
        target = 1.0 / a
        reciprocals = sorted([1.0 / x for x in others], reverse=True)
        n = len(reciprocals)
        # Check all subsets via bitmask
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

def _greedy_build(order):
    """Build a valid set greedily following the given order."""
    S = []
    for x in order:
        candidate = S + [x]
        if _is_valid_fast(candidate):
            S = candidate
    return S

def construct():
    start_time = time.time()
    deadline = 80.0  # safety margin before 90s kill

    best = list(range(13, 25))  # fallback: 12 elements

    # Try multiple greedy orderings
    base = list(range(1, 25))

    orders = []
    # Descending
    orders.append(sorted(base, reverse=True))
    # Ascending
    orders.append(sorted(base))
    # Even first then odd
    orders.append([x for x in base if x % 2 == 0] + [x for x in base if x % 2 == 1])
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

    for order in orders:
        if time.time() - start_time > deadline:
            break
        S = _greedy_build(order)
        if len(S) > len(best):
            best = S

    # Local search: try adding elements not in best
    if time.time() - start_time > deadline:
        return best

    S = list(best)
    improved = True
    while improved and time.time() - start_time < deadline:
        improved = False
        for x in range(1, 25):
            if x in S:
                continue
            candidate = S + [x]
            if _is_valid_fast(candidate):
                S = candidate
                improved = True
                break

    # Try removing one element and adding two
    if time.time() - start_time > deadline:
        return S

    best2 = list(S)
    for i in range(len(S)):
        if time.time() - start_time > deadline:
            break
        for j in range(i + 1, len(S)):
            if time.time() - start_time > deadline:
                break
            reduced = [x for k, x in enumerate(S) if k != i and k != j]
            for x in range(1, 25):
                if x in reduced:
                    continue
                for y in range(1, 25):
                    if y in reduced or y == x:
                        continue
                    candidate = reduced + [x, y]
                    if len(candidate) > len(best2) and _is_valid_fast(candidate):
                        best2 = candidate

    if len(best2) > len(S):
        S = best2

    return S
# EVOLVE-BLOCK-END
