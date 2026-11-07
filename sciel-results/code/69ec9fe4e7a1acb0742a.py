"""Cardinality-separated subset sums: fast incremental bitset-DP oracle + multi-neighborhood local search."""
import time
import random

def _compute_dp(arr):
    """Compute cardinality-indexed bitset DP for arr. dp[k] has bit s set iff sum s achievable with exactly k elements."""
    n = len(arr)
    dp = [0] * (n + 1)
    dp[0] = 1  # empty subset, sum 0
    for x in arr:
        for k in range(n, 0, -1):
            dp[k] |= (dp[k - 1] << x)
    return dp

def _is_valid_dp(dp):
    """Check cross-cardinality collisions in a precomputed dp."""
    seen = 0
    for k in range(len(dp)):
        if dp[k] & seen:
            return False
        seen |= dp[k]
    return True

def _valid_subset_sums(arr):
    """Return True if no sum appears in two different cardinalities (incl. empty subset sum 0)."""
    dp = _compute_dp(arr)
    return _is_valid_dp(dp)

def _try_add_fast(arr, x, dp_cache=None):
    """Return (True, new_dp) if adding x keeps the cardinality-separation property.
    If dp_cache is provided and matches arr's dp, use it for speedup."""
    n = len(arr)
    if dp_cache is not None and len(dp_cache) == n + 1:
        dp = list(dp_cache)
    else:
        dp = _compute_dp(arr)
    # Add x: for k from n+1 down to 1
    new_dp = dp + [0]  # extend to n+2 entries
    for k in range(n + 1, 0, -1):
        new_dp[k] |= (dp[k - 1] << x)
    # Check validity
    if not _is_valid_dp(new_dp):
        return False, None
    return True, new_dp

def _grow(cur, order, deadline):
    """Repeated-pass greedy growth with incremental DP."""
    cur = list(cur)
    cur_set = set(cur)
    dp = _compute_dp(cur)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for x in order:
            if time.time() > deadline:
                break
            if x in cur_set:
                continue
            ok, new_dp = _try_add_fast(cur, x, dp)
            if ok:
                cur.append(x)
                cur_set.add(x)
                dp = new_dp
                improved = True
    return cur

def _swap_search(cur, deadline):
    """Local search with multiple neighborhoods: 1-for-2, 2-for-3, 1-for-1 replacement."""
    cur = list(cur)
    n_max = 256
    improved = True
    while improved and time.time() < deadline:
        improved = False
        n = len(cur)
        
        # Neighborhood 1: 1-for-2 (remove one, add two → net +1)
        for i in range(n):
            if time.time() > deadline:
                return cur
            x = cur[i]
            base = cur[:i] + cur[i+1:]
            base_set = set(base)
            base_dp = _compute_dp(base)
            if not _is_valid_dp(base_dp):
                continue
            for a in range(1, n_max + 1):
                if time.time() > deadline:
                    return cur
                if a == x or a in base_set:
                    continue
                ok1, dp1 = _try_add_fast(base, a, base_dp)
                if not ok1:
                    continue
                base2 = base + [a]
                base2_set = base_set | {a}
                for b in range(a + 1, n_max + 1):
                    if time.time() > deadline:
                        return cur
                    if b == x or b in base2_set:
                        continue
                    ok2, dp2 = _try_add_fast(base2, b, dp1)
                    if ok2:
                        cur = base2 + [b]
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break
        
        # Neighborhood 2: 2-for-3 (remove two, add three → net +1)
        if not improved and time.time() < deadline:
            n = len(cur)
            for i in range(n):
                if time.time() > deadline:
                    return cur
                for j in range(i + 1, n):
                    if time.time() > deadline:
                        return cur
                    x1, x2 = cur[i], cur[j]
                    base = cur[:i] + cur[i+1:j] + cur[j+1:]
                    base_set = set(base)
                    base_dp = _compute_dp(base)
                    if not _is_valid_dp(base_dp):
                        continue
                    for a in range(1, n_max + 1):
                        if time.time() > deadline:
                            return cur
                        if a == x1 or a == x2 or a in base_set:
                            continue
                        ok1, dp1 = _try_add_fast(base, a, base_dp)
                        if not ok1:
                            continue
                        base2 = base + [a]
                        base2_set = base_set | {a}
                        for b in range(a + 1, n_max + 1):
                            if time.time() > deadline:
                                return cur
                            if b == x1 or b == x2 or b in base2_set:
                                continue
                            ok2, dp2 = _try_add_fast(base2, b, dp1)
                            if not ok2:
                                continue
                            base3 = base2 + [b]
                            base3_set = base2_set | {b}
                            for c in range(b + 1, n_max + 1):
                                if time.time() > deadline:
                                    return cur
                                if c == x1 or c == x2 or c in base3_set:
                                    continue
                                ok3, _ = _try_add_fast(base3, c, dp2)
                                if ok3:
                                    cur = base3 + [c]
                                    improved = True
                                    break
                                if time.time() > deadline:
                                    break
                            if improved or time.time() > deadline:
                                break
                        if improved or time.time() > deadline:
                            break
                    if improved or time.time() > deadline:
                        break
                if improved or time.time() > deadline:
                    break
        
        # Neighborhood 3: 1-for-1 replacement (swap one element for another, same size)
        # This helps escape local optima by exploring the same-size neighborhood
        if not improved and time.time() < deadline:
            n = len(cur)
            for i in range(n):
                if time.time() > deadline:
                    return cur
                x = cur[i]
                base = cur[:i] + cur[i+1:]
                base_set = set(base)
                base_dp = _compute_dp(base)
                if not _is_valid_dp(base_dp):
                    continue
                for a in range(1, n_max + 1):
                    if time.time() > deadline:
                        return cur
                    if a == x or a in base_set:
                        continue
                    ok, new_dp = _try_add_fast(base, a, base_dp)
                    if ok:
                        # Same size, but different element — accept if it leads to future growth
                        # Only accept if a > x (prefer larger elements to leave room for smaller)
                        # Actually, accept any valid swap to explore the space
                        cur = base + [a]
                        improved = True
                        break
                if improved:
                    break
    
    return cur

def construct():
    start_time = time.time()
    deadline = start_time + 82  # 8s safety margin under 90s
    rng = random.Random(12345)
    best = []
    best_len = 0

    # Base sets to try
    bases = [
        [1, 2, 4, 8, 16, 32, 64, 128, 256],  # powers of 2
        [1, 3, 7, 15, 31, 63, 127, 255],      # 2^k - 1
        [2, 5, 11, 23, 47, 95, 191],          # 2*2^k + 1 style
        [1, 2, 4, 8, 16, 32, 64, 128, 200],   # mixed
        [1, 2, 4, 8, 16, 32, 64, 128, 192],   # mixed alt
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
    for _ in range(4):
        o = list(candidates_all)
        rng.shuffle(o)
        orders.append(o)

    # Phase 1: repeated-pass greedy from each base x order
    for base in bases:
        if time.time() > deadline - 20:  # reserve time for local search
            break
        if base and not _valid_subset_sums(base):
            continue
        for order in orders:
            if time.time() > deadline - 20:
                break
            cur = _grow(list(base), order, deadline - 20)
            if len(cur) > best_len:
                best = list(cur)
                best_len = len(cur)

    # Phase 2: swap-based local search on the best found (multiple rounds)
    if time.time() < deadline - 5:
        for _ in range(3):  # up to 3 rounds of local search
            if time.time() > deadline - 5:
                break
            before = len(best)
            best = _swap_search(best, deadline - 5)
            if len(best) <= before:
                break
            best_len = len(best)

    # Phase 3: final repeated-pass extension of best with all candidates
    if time.time() < deadline - 2:
        best = _grow(best, candidates_all, deadline - 1)
        best_len = len(best)

    # Phase 4: one more local search round if time permits
    if time.time() < deadline - 1:
        best = _swap_search(best, deadline - 0.5)
        best_len = len(best)

    # Ensure return type is list of Python ints
    return [int(v) for v in best]
