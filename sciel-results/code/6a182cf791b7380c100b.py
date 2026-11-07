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
            
        # Optimization: Check only a subset of candidates if time is tight?
        # No, check all.
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
    best = []
    # Try to find a large consecutive block
    # This is a fast heuristic to get a decent starting point
    for n in range(60, 0, -1):
        for a in range(1, 257 - n + 1):
            arr = list(range(a, a + n))
            if _is_valid(arr):
                if len(arr) > len(best):
                    best = arr
                break
        if len(best) == n:
            break
    return best

def _polish(cur, deadline):
    cur = list(cur)
    n = len(cur)
    improved = True
    while improved and time.monotonic() < deadline:
        improved = False
        cur_set = set(cur)
        # Try to replace one element to allow adding another
        # Or just try to add another by swapping
        for i in range(n):
            if time.monotonic() > deadline:
                break
            old = cur[i]
            remaining = cur[:i] + cur[i+1:]
            remaining_set = set(remaining)
            rem_dp = _init_dp(remaining)
            
            # Try replacing old with v, then see if we can add something
            # Or just check if replacement allows immediate growth
            # To save time, just try replacements that are valid for the remaining set
            # and then try to grow from there
            
            # Heuristic: Try smaller values to pack tighter? Or larger?
            # Let's try a range of candidates
            candidates = [v for v in range(1, 257) if v not in remaining_set and v != old]
            if not candidates:
                continue
            
            # Limit candidates to speed up polish
            # Try smallest few and largest few
            test_candidates = candidates[:10] + candidates[-10:]
            
            for v in test_candidates:
                if time.monotonic() > deadline:
                    break
                # Check if v can be added to remaining
                new_dp = _verify_add(rem_dp, n - 1, v)
                if new_dp is not None:
                    cur_new = remaining + [v]
                    cur_new.sort()
                    # Try to grow this new set
                    grown = _grow_greedy(cur_new, 'smallest', deadline)
                    if len(grown) > len(cur):
                        cur = grown
                        n = len(cur)
                        improved = True
                        break
            if improved:
                break
    return cur

def construct():
    t0 = time.monotonic()
    # Evaluator kills after 90s. Use 85s deadline.
    deadline = t0 + 85.0
    best = None
    
    def update_best(arr):
        nonlocal best
        if best is None or len(arr) > len(best):
            best = list(arr)
    
    # 1. Quick consecutive search
    res = _search_consecutive()
    if res:
        update_best(res)
    
    # 2. Greedy from empty
    if best is None or len(best) < 35:
        res = _grow_greedy([], 'smallest', deadline)
        update_best(res)
        
    # 3. Greedy from powers of 2
    p2 = [1, 2, 4, 8, 16, 32, 64, 128, 256]
    res = _grow_greedy(p2, 'smallest', deadline)
    update_best(res)
    
    # 4. Randomized restarts
    rng = random.Random(42)
    if time.monotonic() < deadline - 5:
        for _ in range(1000):
            if time.monotonic() > deadline - 2:
                break
            k = rng.randint(0, 10)
            if k == 0:
                start = []
            else:
                start = rng.sample(range(1, 257), k)
                if not _is_valid(start):
                    continue
            strategy = rng.choice(['smallest', 'largest', 'random'])
            res = _grow_greedy(start, strategy, deadline, rng)
            update_best(res)
            
    # 5. Polish the best found
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
        
    # Ensure types are correct
    return [int(x) for x in best]
