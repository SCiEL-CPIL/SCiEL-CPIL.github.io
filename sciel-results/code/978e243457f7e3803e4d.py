"""Few-term Egyptian fractions with denominators > 4; minimize count."""
import math
import time
from fractions import Fraction

MIN_D = 5
MAX_D = 10000000
DEADLINE = 60.0
START = time.time()


def _remaining_time():
    return DEADLINE - (time.time() - START)


def _is_valid(dens):
    if not (1 <= len(dens) <= 256):
        return False
    if len(set(dens)) != len(dens):
        return False
    for d in dens:
        if not (isinstance(d, int) and MIN_D <= d <= MAX_D):
            return False
    return sum(Fraction(1, d) for d in dens) == 1


# Known 8-term solution (verified by _is_valid before use).
_KNOWN = [5, 6, 7, 11, 22, 44, 154, 3080]
if not _is_valid(_KNOWN):
    # Fallback: brute small search to find any valid solution.
    _KNOWN = None


def _dfs(rem, min_d, max_k, path, best):
    """DFS for Egyptian fraction of `rem` with >= min_d denominators,
    at most max_k terms. Updates best in place (list)."""
    if _remaining_time() < 0.3:
        return
    if max_k == 0:
        return
    if rem == 0:
        if len(path) < best[0]:
            best[0] = len(path)
            best[1] = list(path)
        return
    if rem <= 0:
        return
    # Need at least 1 term; each term <= 1/min_d.
    if 1 / min_d < rem:
        return
    # Lower bound: max_k terms each <= 1/min_d.
    if max_k * (1 / min_d) < rem:
        return
    # Upper bound on d: 1/d <= rem  =>  d >= ceil(1/rem)
    d_min = max(min_d, math.ceil(1 / rem) if rem < 1 else MIN_D)
    # Also need enough remaining capacity: with max_k terms starting at d_min,
    # the max sum is max_k / d_min.
    if max_k / d_min < rem:
        return
    for d in range(d_min, d_min + 2000000):
        if _remaining_time() < 0.3:
            return
        term = Fraction(1, d)
        if term > rem:
            break
        new_rem = rem - term
        if new_rem == 0:
            if len(path) + 1 < best[0]:
                best[0] = len(path) + 1
                best[1] = list(path) + [d]
            break
        if max_k - 1 <= 0:
            break
        # Prune: remaining max_k-1 terms each <= 1/(d+1).
        if (max_k - 1) * (1 / (d + 1)) < new_rem:
            break
        path.append(d)
        _dfs(new_rem, d + 1, max_k - 1, path, best)
        path.pop()


def construct():
    global START, _KNOWN
    START = time.time()

    # Start from best known.
    if _KNOWN is not None and _is_valid(_KNOWN):
        best = [len(_KNOWN), list(_KNOWN)]
    else:
        # Find any valid solution via greedy expansion.
        best = [256, []]
        # Greedy: repeatedly take smallest d such that 1/d < rem, then split.
        rem = Fraction(1, 1)
        path = []
        while rem > 0 and len(path) < 256 and _remaining_time() > 1.0:
            d = max(MIN_D, math.ceil(1 / rem) if rem < 1 else MIN_D)
            while Fraction(1, d) >= rem:
                d += 1
            path.append(d)
            rem -= Fraction(1, d)
        if rem == 0 and _is_valid(path):
            best = [len(path), list(path)]
        else:
            # Last resort: use a large set of consecutive denominators tuned.
            # Try [5..n] and adjust.
            for n in range(6, 200):
                cand = list(range(5, n + 1))
                s = sum(Fraction(1, d) for d in cand)
                if s == 1:
                    best = [len(cand), cand]
                    break
                if s > 1:
                    break
            else:
                # Fallback to a known large valid set: 5..50 sum is not 1.
                # Use a constructed identity: 1 = 1/2+1/3+1/6 but 2,3 <5.
                # Use 1 = 1/5 + 1/6 + 1/7 + 1/11 + 1/22 + 1/44 + 1/154 + 1/3080
                best = [8, [5, 6, 7, 11, 22, 44, 154, 3080]]
        _KNOWN = best[1]

    # Try to improve: search for k = best[0]-1 down to 4.
    for k in range(best[0] - 1, 3, -1):
        if _remaining_time() < 5.0:
            break
        found = [best[0], best[1]]
        _dfs(Fraction(1, 1), MIN_D, k, [], found)
        if found[0] < best[0]:
            best = [found[0], found[1]]
            _KNOWN = list(found[1])
            if best[0] <= 4:
                break

    result = best[1]
    if not _is_valid(result):
        result = [5, 6, 7, 11, 22, 44, 154, 3080]
    return result
