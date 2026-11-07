import time
import random

def _build_dp(arr):
    """Build cardinality-indexed subset sum bitsets for arr."""
    n = len(arr)
    dp = [0] * (n + 1)
    dp[0] = 1
    for x in arr:
        for k in range(n, 0, -1):
            dp[k] |= (dp[k - 1] << x)
    return dp

def _check_dp(dp):
    """Check if no two different cardinalities share a sum."""
    acc = 0
    for k in range(len(dp)):
        if dp[k] & acc:
            return False
        acc |= dp[k]
    return True

def _verify(arr):
    n = len(arr)
    if n < 1 or n > 60:
        return False
    s = set()
    for v in arr:
        if not isinstance(v, int) or isinstance(v, bool):
            return False
        if v < 1 or v > 256:
            return False
        if v in s:
            return False
        s.add(v)
    
    dp = _build_dp(arr)
    return _check_dp(dp)

def _try_add(arr, dp, v, n):
    """Try adding v to arr with existing dp. Returns new dp or None if invalid."""
    # Extend dp to size n+2
    new_dp = dp + [0] * (len(dp) - n + 1)
    # Actually, we need to be careful: dp has n+1 entries for n elements
    # After adding, we need n+2 entries
    new_dp = [0] * (n + 2)
    for k in range(n + 1):
        new_dp[k] = dp[k]
    # Update in reverse
    for k in range(n + 1, 0, -1):
        new_dp[k] |= (dp[k - 1] << v)
    if not _check_dp(new_dp):
        return None
    return new_dp

def _grow(start, deadline):
    """Grow a valid set by incrementally adding elements."""
    cur = list(start)
    cur_set = set(cur)
    n = len(cur)
    dp = _build_dp(cur)
    
    # Precompute which values might be safe to try
    # Larger values are less likely to cause sum collisions
    candidates = [v for v in range(1, 257) if v not in cur_set]
    # Sort by descending value - larger values tend to create sums in higher ranges
    candidates.sort(reverse=True)
    
    improved = True
    while improved and time.monotonic() < deadline:
        improved = False
        n = len(cur)
        for v in candidates:
            if v in cur_set:
                continue
            new_dp = _try_add(cur, dp, v, n)
            if new_dp is not None:
                cur.append(v)
                cur_set.add(v)
                dp = new_dp
                improved = True
                break
    return cur

def construct():
    t0 = time.monotonic()
    deadline = t0 + 85.0
    best = None
    best_len = 0

    def consider(arr):
        nonlocal best, best_len
        if len(arr) > best_len:
            if _verify(arr):
                best = list(arr)
                best_len = len(arr)

    # 1) Try consecutive runs from high end, largest to smallest
    for n in range(60, 0, -1):
        a = 257 - n
        arr = list(range(a, 257))
        if _verify(arr):
            best = arr
            best_len = n
            break
        if time.monotonic() > deadline - 10:
            break

    # 2) Greedy grow from best consecutive
    if best is not None:
        g = _grow(best, deadline)
        if len(g) > best_len:
            best = g
            best_len = len(g)

    # 3) Random restarts with growth
    rng = random.Random(12345)
    attempts = 0
    while time.monotonic() < deadline - 3 and attempts < 30000:
        attempts += 1
        n = rng.randint(15, 55)
        # Mix low and high values to reduce sum collisions
        low = list(range(1, 100))
        high = list(range(160, 257))
        mid = list(range(100, 160))
        
        # Weighted sampling: more from extremes
        half = n // 2
        low_count = min(half, len(low))
        high_count = min(n - low_count, len(high))
        mid_count = min(n - low_count - high_count, len(mid))
        
        pool = rng.sample(low, low_count) + rng.sample(high, high_count)
        if mid_count > 0:
            pool += rng.sample(mid, mid_count)
        
        if len(pool) < n:
            continue
        arr = rng.sample(pool, n)
        
        # Quick check before full verify
        if _verify(arr):
            g = _grow(arr, deadline)
            if len(g) > best_len:
                best = g
                best_len = len(g)
            if best_len >= 60:
                break

    # 4) Final fallback
    if best is None:
        best = list(range(226, 257))

    best = [int(v) for v in best]
    return best
