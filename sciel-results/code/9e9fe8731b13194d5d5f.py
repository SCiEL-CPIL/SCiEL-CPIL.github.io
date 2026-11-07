"""Cardinality-separated subset sums: greedy growth with incremental bitset DP."""
import time

def _valid_subset_sums(arr):
    """Return True if no sum appears in two different cardinalities."""
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
    new_arr = arr + [x]
    return _valid_subset_sums(new_arr)

def construct():
    deadline = time.time() + 75  # 75s guard, 15s safety margin
    best = []

    # Try multiple starting seeds and growth strategies
    # Strategy: start with a base set, then greedily add candidates
    # Candidates tried in multiple orders to explore different basins

    # Base sets to try
    bases = [
        [1, 2, 4, 8, 16, 32, 64, 128],  # powers of 2
        [1, 3, 7, 15, 31, 63, 127, 255],  # 2^k - 1
        [2, 5, 11, 23, 47, 95, 191],  # 2*2^k + 1 style
        [1, 2, 4, 8, 16, 32, 64, 128, 200],  # mixed
    ]

    # Candidate orders: ascending, descending, and several interleavings
    candidates_all = list(range(1, 257))
    orders = [
        candidates_all,                          # ascending
        list(reversed(candidates_all)),          # descending
        # Even then odd
        [i for i in range(1, 257, 2)] + [i for i in range(2, 257, 2)],
        # Large first, then small
        list(range(256, 0, -1)),
        # Alternating from both ends
        [],
    ]
    # Build the alternating order
    lo, hi = 1, 256
    alt = []
    while lo <= hi:
        alt.append(lo)
        if lo != hi:
            alt.append(hi)
        lo += 1
        hi -= 1
    orders.append(alt)

    for base in bases:
        if time.time() > deadline:
            break
        # Verify base
        if not _valid_subset_sums(base):
            continue
        current = list(base)
        for order in orders:
            if time.time() > deadline:
                break
            cur = list(current)
            for x in order:
                if time.time() > deadline:
                    break
                if x in cur:
                    continue
                if _try_add(cur, x):
                    cur.append(x)
            if len(cur) > len(best):
                best = list(cur)
        # Also try greedy from this base with each order
        for order in orders:
            if time.time() > deadline:
                break
            cur = list(base)
            for x in order:
                if time.time() > deadline:
                    break
                if x in cur:
                    continue
                if _try_add(cur, x):
                    cur.append(x)
            if len(cur) > len(best):
                best = list(cur)

    # Also try pure greedy from empty with each order
    for order in orders:
        if time.time() > deadline:
            break
        cur = []
        for x in order:
            if time.time() > deadline:
                break
            if _try_add(cur, x):
                cur.append(x)
        if len(cur) > len(best):
            best = list(cur)

    # Final: try to extend best further with remaining candidates
    if time.time() < deadline - 5:
        for x in range(1, 257):
            if time.time() > deadline:
                break
            if x in best:
                continue
            if _try_add(best, x):
                best.append(x)

    # Ensure return type is list of Python ints
    return [int(v) for v in best]
