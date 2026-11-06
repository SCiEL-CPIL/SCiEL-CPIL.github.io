"""
Structural change: instead of a single 0/1 DP pass that only finds *a* minimum-count
solution (and whose reconstruction can be ambiguous), we compute dp[s] = min count
for every sum s in [0, target] using a clean 0/1 knapsack over the divisors of 5040.
Then we reconstruct the lexicographically-smallest (in terms of divisor value order)
minimum-count subset by walking the DP table.  The key structural improvement over
the seed is:

  1. We precompute the full dp table (min count to reach each sum) once.
  2. Reconstruction uses the dp table directly: at each step we choose the smallest
     divisor d such that dp[remaining - d] == dp[remaining] - 1, guaranteeing we
     always stay on a minimum-count path and obtain a deterministic, valid answer.
  3. We add a verification step and a fallback BFS-by-count for safety.

This is still O(num_divisors * target) ≈ 48 * 4979 ≈ 240k operations, well within budget.
"""
import time

_DEADLINE = 80.0  # safety margin under 90 s kill


def _all_divisors(n):
    divs = []
    i = 1
    while i * i <= n:
        if n % i == 0:
            divs.append(i)
            if i != n // i:
                divs.append(n // i)
        i += 1
    divs.sort()
    return divs


def _verify(chosen, n, target):
    if len(chosen) > 256:
        return False
    seen = set()
    total = 0
    for d in chosen:
        if d in seen:
            return False
        if d <= 0 or n % d != 0:
            return False
        seen.add(d)
        total += d
    return total == target


def _bfs_by_count(divs, target, n, deadline):
    """Fallback: try all subsets of size k for k = 1, 2, ..."""
    from itertools import combinations
    divs_sorted = sorted(divs, reverse=True)
    for k in range(1, 257):
        if time.time() > deadline:
            break
        # Prune: if sum of k largest < target, no solution of this size
        if sum(divs_sorted[:k]) < target:
            continue
        for combo in combinations(divs_sorted, k):
            if time.time() > deadline:
                break
            if sum(combo) == target:
                return list(combo)
    return []


def construct():
    start = time.time()
    target = 4979
    n = 5040

    divs = _all_divisors(n)
    # Keep only divisors <= target (larger ones can't be used)
    divs = [d for d in divs if d <= target]
    num_divs = len(divs)

    # Quick check: is target itself a divisor?
    if n % target == 0:
        return [target]

    # ---- DP: dp[s] = minimum number of divisors needed to sum to s ----
    INF = 999999
    dp = [INF] * (target + 1)
    dp[0] = 0

    for d in divs:
        if time.time() > _DEADLINE:
            break
        # 0/1 knapsack: iterate descending
        for s in range(target, d - 1, -1):
            if dp[s - d] + 1 < dp[s]:
                dp[s] = dp[s - d] + 1

    best_count = dp[target]
    if best_count >= INF:
        # No solution found by DP (shouldn't happen, but fallback)
        result = _bfs_by_count(divs, target, n, _DEADLINE)
        if result:
            return result
        # Last-resort greedy
        remainder = target
        chosen = []
        for d in sorted(divs, reverse=True):
            if d <= remainder:
                chosen.append(d)
                remainder -= d
            if remainder == 0:
                break
        return chosen

    # ---- Reconstruction using the DP table ----
    # We want a valid minimum-count subset.  Walk from target down to 0,
    # at each step picking the smallest divisor d such that:
    #   dp[remaining] == dp[remaining - d] + 1
    # This guarantees we stay on a shortest path.
    chosen = []
    remaining = target
    used = set()
    # Sort divisors ascending so we prefer smaller divisors (deterministic, valid)
    divs_asc = sorted(divs)

    while remaining > 0:
        if time.time() > _DEADLINE:
            break
        found = False
        for d in divs_asc:
            if d in used:
                continue
            if d > remaining:
                break
            if dp[remaining - d] == dp[remaining] - 1:
                chosen.append(d)
                used.add(d)
                remaining -= d
                found = True
                break
        if not found:
            # Should not happen if DP is correct, but break to avoid infinite loop
            break

    if _verify(chosen, n, target):
        return chosen

    # Fallback: BFS by count
    result = _bfs_by_count(divs, target, n, _DEADLINE)
    if result:
        return result

    # Last resort: greedy
    remainder = target
    chosen = []
    for d in sorted(divs, reverse=True):
        if d <= remainder:
            chosen.append(d)
            remainder -= d
        if remainder == 0:
            break
    return chosen
