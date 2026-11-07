"""Cardinality-separated subset sums: fast bitset-DP oracle + repeated-pass greedy + swap local search."""
import time
import random

def _valid_subset_sums(arr):
    """Return True if no sum appears in two different cardinalities (incl. empty subset sum 0)."""
    n = len(arr)
    max_sum = sum(arr)
    # dp[k] is a Python int bitset: bit s set means sum s achievable with exactly k elements
    dp = [0] * (n + 1)
    dp[0] = 1  # empty subset, sum 0
    for x in arr:
        for k in range(n, 0, -1):
            dp[k] |= (dp[k - 1] << x)
    # Check cross-cardinality collisions
    seen = 0
    for k in range(n + 1):
        if dp[k] & seen:
            return False
        seen |= dp[k]
    return True

def _try_add(arr, x):
    """Return True if adding x keeps the cardinality-separation property."""
    return _valid_subset_sums(arr + [x])

def _grow(cur, order, deadline):
    """Repeated-pass greedy growth: scan candidates, add any that fit, until a full pass adds nothing."""
    cur = list(cur)
    cur_set = set(cur)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for x in order:
            if time.time() > deadline:
                break
            if x in cur_set:
                continue
            if _try_add(cur, x):
                cur.append(x)
                cur_set.add(x)
                improved = True
    return cur

def _swap_search(cur, deadline):
    """Local search: try replacing one element with two new ones (net +1) if valid. Repeat until no gain."""
    cur = list(cur)
    n_max = 256
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i, x in enumerate(cur):
            if time.time() > deadline:
                return cur
            # candidate removal
            base = cur[:i] + cur[i+1:]
            # try adding two distinct new elements a < b
            for a in range(1, n_max + 1):
                if time.time() > deadline:
                    return cur
                if a == x or a in base:
                    continue
                if not _try_add(base, a):
                    continue
                base2 = base + [a]
                for b in range(a + 1, n_max + 1):
                    if time.time() > deadline:
                        return cur
                    if b == x or b in base2:
                        continue
                    if _try_add(base2, b):
                        new_cur = base2 + [b]
                        cur = new_cur
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break
    return cur

def construct():
    deadline = time.time() + 75  # safety margin under 90s kill
    rng = random.Random(12345)
    best = []

    # Base sets to try
    bases = [
        [1, 2, 4, 8, 16, 32, 64, 128, 256],  # powers of 2 (9 elements)
        [1, 3, 7, 15, 31, 63, 127, 255],      # 2^k - 1
        [2, 5, 11, 23, 47, 95, 191],          # 2*2^k + 1 style
        [1, 2, 4, 8, 16, 32, 64, 128, 200],   # mixed
        [],                                    # pure greedy from empty
    ]

    # Candidate orderings
    candidates_all = list(range(1, 257))
    orders = [
        candidates_all,
        list(reversed(candidates_all)),
        [i for i in range(1, 257, 2)] + [i for i in range(2, 257, 2)],
        list(range(256, 0, -1)),
    ]
    # Alternating from both ends
    lo, hi = 1, 256
    alt = []
    while lo <= hi:
        alt.append(lo)
        if lo != hi:
            alt.append(hi)
        lo += 1
        hi -= 1
    orders.append(alt)
    # A few shuffled orders for diversification
    for _ in range(3):
        o = list(candidates_all)
        rng.shuffle(o)
        orders.append(o)

    # Phase 1: repeated-pass greedy from each base x order
    for base in bases:
        if time.time() > deadline:
            break
        if base and not _valid_subset_sums(base):
            continue
        for order in orders:
            if time.time() > deadline:
                break
            cur = _grow(list(base), order, deadline)
            if len(cur) > len(best):
                best = list(cur)

    # Phase 2: swap-based local search on the best found
    if time.time() < deadline - 10:
        improved = True
        while improved and time.time() < deadline - 5:
            before = len(best)
            best = _swap_search(best, deadline - 5)
            improved = len(best) > before

    # Phase 3: final repeated-pass extension of best with all candidates
    if time.time() < deadline - 3:
        best = _grow(best, candidates_all, deadline - 1)

    # Ensure return type is list of Python ints
    return [int(v) for v in best]
