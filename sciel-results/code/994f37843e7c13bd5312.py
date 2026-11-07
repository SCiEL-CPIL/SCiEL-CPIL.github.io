import time
import random

def _verify(arr):
    # arr: iterable of ints
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
    # bitset DP by cardinality
    dp = [0] * (n + 1)
    dp[0] = 1
    for x in arr:
        for k in range(n, 0, -1):
            dp[k] |= (dp[k - 1] << x)
    acc = 0
    for k in range(n + 1):
        if dp[k] & acc:
            return False
        acc |= dp[k]
    return True

def _grow(start, deadline):
    cur = list(start)
    cur_set = set(cur)
    n = len(cur)
    dp = [0] * (n + 1)
    dp[0] = 1
    for x in cur:
        for k in range(n, 0, -1):
            dp[k] |= (dp[k - 1] << x)
    acc = 0
    for k in range(n + 1):
        acc |= dp[k]
    improved = True
    while improved and time.monotonic() < deadline:
        improved = False
        for v in range(1, 257):
            if v in cur_set:
                continue
            new_acc = 1
            valid = True
            for k in range(1, n + 2):
                base = dp[k] if k <= n else 0
                shifted = dp[k-1] << v
                new_dp_k = base | shifted
                if new_dp_k & new_acc:
                    valid = False
                    break
                new_acc |= new_dp_k
            if valid:
                cur.append(v)
                cur_set.add(v)
                n += 1
                new_dp = [0] * (n + 1)
                new_dp[0] = 1
                for k in range(n, 0, -1):
                    old_k = dp[k] if k <= n - 1 else 0
                    new_dp[k] = old_k | (dp[k-1] << v)
                dp = new_dp
                acc = new_acc
                improved = True
                break
    return cur

def construct():
    t0 = time.monotonic()
    deadline = t0 + 85.0  # safety margin under 90s wall-clock kill
    best = None

    def consider(arr):
        nonlocal best
        if best is None or len(arr) > len(best):
            if _verify(arr):
                best = list(arr)

    # 1) consecutive runs (high end) - fast structured family
    for n in range(60, 0, -1):
        a = 257 - n
        arr = list(range(a, 257))
        if _verify(arr):
            best = arr
            break  # largest n consecutive that works; smaller n also work but we want max
        if time.monotonic() > deadline - 2:
            break

    # 2) greedy grow from best consecutive
    if best is not None:
        g = _grow(best, deadline)
        if len(g) > len(best):
            best = g

    # 3) random-restart + grow, bimodal starts, until deadline
    rng = random.Random(12345)
    attempts = 0
    while time.monotonic() < deadline - 1 and attempts < 20000:
        attempts += 1
        n = rng.randint(20, 60)
        # bimodal pool to encourage gap
        low = list(range(1, 121))
        high = list(range(137, 257))
        pool = rng.sample(low, min(n//2, len(low))) + rng.sample(high, n - min(n//2, len(low)))
        if len(pool) < n:
            continue
        arr = rng.sample(pool, n)
        if _verify(arr):
            g = _grow(arr, deadline)
            if best is None or len(g) > len(best):
                best = g
            if best is not None and len(best) >= 60:
                break

    if best is None:
        # absolute safe fallback: consecutive 226..256 (31 elements)
        best = list(range(226, 257))
        _verify(best)  # ensure
    # ensure python int list
    best = [int(v) for v in best]
    return best
