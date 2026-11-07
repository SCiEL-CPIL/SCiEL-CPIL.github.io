import time
import random

def _init_dp(arr):
    n = len(arr)
    if n == 0:
        return [1]
    dp = [0] * (n + 1)
    dp[0] = 1
    for x in arr:
        for k in range(n, 0, -1):
            dp[k] |= (dp[k - 1] << x)
    return dp

def _is_valid(arr):
    n = len(arr)
    if n == 0:
        return True
    dp = _init_dp(arr)
    acc = 0
    for k in range(n + 1):
        if dp[k] & acc:
            return False
        acc |= dp[k]
    return True

def _verify_add(dp, n, x):
    new_dp = [0] * (n + 2)
    for k in range(n + 1):
        new_dp[k] = dp[k]
    for k in range(n + 1, 0, -1):
        new_dp[k] |= (dp[k - 1] << x)
    acc = 0
    for k in range(n + 2):
        if new_dp[k] & acc:
            return None
        acc |= new_dp[k]
    return new_dp

def _grow_greedy(start, strategy='smallest', deadline=None, rng=None):
    cur = list(start)
    cur_set = set(cur)
    n = len(cur)
    dp = _init_dp(cur)
    
    improved = True
    while improved:
        if deadline is not None and time.monotonic() > deadline:
            break
        improved = False
        candidates = [v for v in range(1, 257) if v not in cur_set]
        if not candidates:
            break
        if strategy == 'smallest':
            order = candidates
        elif strategy == 'largest':
            order = list(reversed(candidates))
        elif strategy == 'random':
            order = candidates[:]
            if rng:
                rng.shuffle(order)
            else:
                random.shuffle(order)
        else:
            order = candidates
        for v in order:
            new_dp = _verify_add(dp, n, v)
            if new_dp is not None:
                cur.append(v)
                cur_set.add(v)
                n += 1
                dp = new_dp
                improved = True
                break
    return cur

def _search_consecutive():
    for n in range(60, 0, -1):
        max_k = (n + 1) // 2
        min_required_a = 0
        for k in [max_k, max_k + 1]:
            if 1 <= k <= n:
                val = (k - 1) * (n - k)
                if val > min_required_a:
                    min_required_a = val
        a_min = min_required_a + 1
        a_max = 256 - n + 1
        if a_min <= a_max:
            return list(range(a_min, a_min + n))
    return None

def _polish(cur, deadline):
    cur = list(cur)
    n = len(cur)
    improved = True
    while improved and time.monotonic() < deadline:
        improved = False
        cur_set = set(cur)
        for i in range(n):
            old = cur[i]
            remaining = cur[:i] + cur[i+1:]
            remaining_set = set(remaining)
            rem_dp = _init_dp(remaining)
            # Try replacing with smaller unused values
            for v in range(1, old):
                if v in remaining_set:
                    continue
                new_dp = _verify_add(rem_dp, n - 1, v)
                if new_dp is not None:
                    cur_new = remaining + [v]
                    cur_new.sort()
                    grown = _grow_greedy(cur_new, 'smallest', deadline)
                    if len(grown) > len(cur):
                        cur = grown
                        n = len(cur)
                        improved = True
                        break
            if improved:
                break
    return cur

def _try_grow_from_consecutive(consec, deadline):
    """Try to grow beyond the consecutive set by inserting new elements."""
    cur = list(consec)
    cur_set = set(cur)
    n = len(cur)
    dp = _init_dp(cur)
    
    # Try all possible insertions in different orders
    for strategy in ['smallest', 'largest']:
        if time.monotonic() > deadline:
            break
        improved = True
        while improved:
            if time.monotonic() > deadline:
                break
            improved = False
            candidates = [v for v in range(1, 257) if v not in cur_set]
            if not candidates:
                break
            if strategy == 'smallest':
                order = candidates
            else:
                order = list(reversed(candidates))
            for v in order:
                new_dp = _verify_add(dp, n, v)
                if new_dp is not None:
                    cur.append(v)
                    cur_set.add(v)
                    n += 1
                    dp = new_dp
                    improved = True
                    break
    return cur

def construct():
    t0 = time.monotonic()
    deadline = t0 + 88.0  # Safety margin for 90s limit
    best = None
    
    def update_best(arr):
        nonlocal best
        if best is None or len(arr) > len(best):
            best = list(arr)
    
    # 1. Analytical consecutive search - fast and guaranteed valid
    res = _search_consecutive()
    if res:
        update_best(res)
    
    # 2. Try to grow from consecutive solution
    if best is not None and time.monotonic() < deadline - 10:
        res = _try_grow_from_consecutive(best, deadline)
        update_best(res)
    
    # 3. Greedy searches to try to beat the consecutive result
    if best is None or len(best) < 35:
        res = _grow_greedy([], 'smallest', deadline)
        update_best(res)
        res = _grow_greedy([], 'largest', deadline)
        update_best(res)
        
        p2 = [1, 2, 4, 8, 16, 32, 64, 128, 256]
        res = _grow_greedy(p2, 'smallest', deadline)
        update_best(res)
        res = _grow_greedy(p2, 'largest', deadline)
        update_best(res)
        
        # Randomized restarts
        rng = random.Random(42)
        if time.monotonic() < deadline - 5:
            for _ in range(100):
                if time.monotonic() > deadline - 2:
                    break
                k = rng.randint(0, 5)
                if k == 0:
                    start = []
                else:
                    start = rng.sample(range(1, 257), k)
                    if not _is_valid(start):
                        continue
                strategy = rng.choice(['smallest', 'largest', 'random'])
                res = _grow_greedy(start, strategy, deadline, rng)
                update_best(res)
    
    # 4. Polish the best result
    if best is not None and time.monotonic() < deadline - 3:
        polished = _polish(best, deadline)
        if len(polished) > len(best):
            best = polished
    
    # Final validity check
    if best is not None:
        if not _is_valid(best):
            best = [1, 2, 4, 8, 16, 32, 64, 128, 256]
    
    if best is None:
        best = [1, 2, 4, 8, 16, 32, 64, 128, 256]
    
    return best
