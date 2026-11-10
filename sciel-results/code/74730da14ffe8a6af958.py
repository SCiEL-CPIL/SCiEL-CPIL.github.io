import time
import itertools
from math import gcd

# EVOLVE-BLOCK-START
def _is_valid(elements):
    """Check: distinct, in 1..256, all nonempty subset sums unique,
    and no two distinct subset sums divide one another."""
    n = len(elements)
    if n < 1 or n > 30:
        return False
    if any(not isinstance(e, int) or isinstance(e, bool) for e in elements):
        return False
    if len(set(elements)) != n:
        return False
    if any(e < 1 or e > 256 for e in elements):
        return False

    # Enumerate all nonempty subset sums (n <= 30 -> up to 2^30, but we
    # only use this for small n in search; the final answer is n=8).
    sums = []
    seen = set()
    for mask in range(1, 1 << n):
        s = 0
        for i in range(n):
            if mask & (1 << i):
                s += elements[i]
        if s in seen:
            return False
        seen.add(s)
        sums.append(s)

    # Divisibility check among distinct sums
    uniq = sorted(seen)
    for i in range(len(uniq)):
        s = uniq[i]
        # any multiple of s greater than s and in uniq is a violation
        for j in range(i + 1, len(uniq)):
            t = uniq[j]
            if t % s == 0:
                return False
    return True


def _greedy_superincreasing():
    """Build a superincreasing set of primes in 1..256 with divisibility-free
    subset sums. Greedy: pick the largest feasible prime at each step."""
    primes = [p for p in range(2, 257)
              if all(p % d for d in range(2, int(p ** 0.5) + 1))]
    chosen = []
    prefix = 0
    # We want many elements; try to extend greedily.
    # At each step, pick the smallest prime > prefix (to leave room for more),
    # but also ensure divisibility safety.
    for p in primes:
        if p <= prefix:
            continue
        # Try adding p
        cand = chosen + [p]
        if _is_valid(cand):
            chosen = cand
            prefix += p
            if len(chosen) >= 30:
                break
    return chosen


def construct():
    start = time.time()
    deadline = start + 50.0  # safety margin before 90s kill

    # Primary: a known-good superincreasing prime set (size 8).
    base = [2, 3, 5, 11, 23, 47, 97, 191]
    best = base if _is_valid(base) else []

    # Try to improve via greedy superincreasing construction.
    if time.time() < deadline:
        g = _greedy_superincreasing()
        if len(g) > len(best) and _is_valid(g):
            best = g

    # Local search: try to add one more element to best.
    if time.time() < deadline and len(best) < 30:
        prefix = sum(best)
        for p in range(1, 257):
            if p in best:
                continue
            cand = sorted(best + [p])
            if _is_valid(cand):
                best = cand
                prefix = sum(best)
                break

    # Ensure type correctness: list of plain ints.
    return [int(x) for x in best]
# EVOLVE-BLOCK-END
