import time
import random

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

    # Bitset DP
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

def _is_valid_dp(dp):
    acc = 0
    for mask in dp:
        if mask & acc:
            return False
        acc |= mask
    return True

def _grow(start, deadline, order='desc'):
    cur = list(start)
    cur_set = set(cur)
    n = len(cur)

    # Initialize DP for current set
    dp = [0] * (n + 1)
    dp[0] = 1
    for x in cur:
        for k in range(n, 0, -1):
            dp[k] |= (dp[k - 1] << x)

    improved = True
    while improved and time.monotonic() < deadline:
        improved = False
        n = len(cur)

        # Determine candidate order
        if order == 'desc':
            candidates = range(256, 0, -1)
        else:
            candidates = range(1, 257)

        for v in candidates:
            if v in cur_set:
                continue

            # Test adding v
            new_dp = [0] * (n + 2)
            for k in range(n + 1):
                new_dp[k] = dp[k]
            for k in range(n + 1, 0, -1):
                new_dp[k] |= (dp[k - 1] << v)

            if _is_valid_dp(new_dp):
                cur.append(v)
                cur_set.add(v)
                dp = new_dp
                improved = True
                break
    return cur

def construct():
    t0 = time.monotonic()
    deadline = t0 + 84.0
    best = None

    def consider(arr):
        nonlocal best
        if best is None or len(arr) > len(best):
            if _verify(arr):
                best = list(arr)

    # 1) Try consecutive runs from high end
    for n in range(60, 0, -1):
        a = 257 - n
        arr = list(range(a, 257))
        if _verify(arr):
            best = arr
            break
        if time.monotonic() > deadline - 5:
            break

    # 2) Greedy grow from best consecutive (Descending order)
    if best is not None:
        g = _grow(best, deadline, order='desc')
        if len(g) > len(best):
            best = g

    # 3) Random restarts with growth (Descending order)
    rng = random.Random(424242)
    attempts = 0
    while time.monotonic() < deadline - 3 and attempts < 50000:
        attempts += 1
        n = rng.randint(15, 60)
        # Mix strategies: sometimes pure high, sometimes mixed
        if rng.random() < 0.5:
            # Pure high range
            pool = list(range(100, 257))
            arr = rng.sample(pool, min(n, len(pool)))
        else:
            # Bimodal
            low = list(range(1, 101))
            high = list(range(160, 257))
            half = n // 2
            pool = rng.sample(low, min(half, len(low))) + rng.sample(high, n - min(half, len(low)))
            if len(pool) < n:
                continue
            arr = rng.sample(pool, n)

        if len(arr) < n:
            continue
        if _verify(arr):
            g = _grow(arr, deadline, order='desc')
            if best is None or len(g) > len(best):
                best = g
            if best is not None and len(best) >= 60:
                break

    # 4) Fallback
    if best is None:
        best = list(range(226, 257))

    best = [int(v) for v in best]
    return best
