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
    # Max sum is 60 * 256 = 15360
    # We use a list of ints as bitsets
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

def _verify_fast(arr):
    """Faster verification using Python's arbitrary precision integers."""
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

def _grow_greedy(best, deadline):
    """Try to grow the set greedily by adding one element at a time."""
    cur = list(best)
    cur_set = set(cur)
    improved = True
    while improved and time.monotonic() < deadline:
        improved = False
        # Try adding each unused value from high to low
        for v in range(256, 0, -1):
            if v in cur_set:
                continue
            cand = cur + [v]
            if _verify_fast(cand):
                cur = cand
                cur_set.add(v)
                improved = True
                break
    return cur

def _swap_improve(cur, deadline):
    """Try 1-for-1 swaps to escape local optima, then regrow."""
    improved = True
    while improved and time.monotonic() < deadline:
        improved = False
        n = len(cur)
        cur_set = set(cur)
        # Try each element for removal
        for i in range(n):
            if time.monotonic() > deadline:
                break
            removed = cur[i]
            cur_set.discard(removed)
            base = cur[:i] + cur[i+1:]
            # Try adding each unused value
            for v in range(256, 0, -1):
                if v in cur_set:
                    continue
                cand = base + [v]
                if _verify_fast(cand):
                    # Check if this swap actually helps (we'll regrow later)
                    cur = cand
                    cur_set.discard(removed)
                    cur_set.add(v)
                    improved = True
                    break
            if improved:
                break
    return cur

def _random_search(target_size, deadline, rng):
    """Random search for a set of target_size."""
    attempts = 0
    while time.monotonic() < deadline and attempts < 20000:
        attempts += 1
        # Different strategies for pool selection
        r = rng.randint(0, 3)
        if r == 0:
            pool = list(range(1, 257))
        elif r == 1:
            pool = list(range(80, 257))
        elif r == 2:
            pool = list(range(120, 257))
        else:
            pool = list(range(160, 257))
            
        if len(pool) < target_size:
            continue
            
        cand = rng.sample(pool, target_size)
        if _verify_fast(cand):
            return cand
    return None

def construct():
    t0 = time.monotonic()
    deadline = t0 + 85.0
    rng = random.Random(12345)
    
    # Start with a known valid set of size 31
    best = list(range(226, 257))  # Size 31
    
    # First, try to grow from 31 to 32, 33, ... using greedy
    # But first, let's try random search for larger sizes
    
    # Try random search for size 32
    for target in range(32, 61):
        if time.monotonic() > deadline - 5.0:
            break
        cand = _random_search(target, min(deadline - 2.0, t0 + (target - 31) * 2.0), rng)
        if cand is not None:
            best = cand
            # Try to grow further
            best = _grow_greedy(best, deadline - 2.0)
            break
    
    # If we found something, try to grow it further
    if len(best) > 31:
        # Try swap improvements
        if time.monotonic() < deadline - 3.0:
            improved = _swap_improve(best, deadline - 3.0)
            if len(improved) > len(best):
                best = improved
        # Final greedy growth
        if time.monotonic() < deadline - 1.0:
            best = _grow_greedy(best, deadline - 1.0)
    else:
        # Still at 31, try harder
        # Try structured approaches
        # 1. Try sets with larger gaps
        if time.monotonic() < deadline - 5.0:
            # Try random search with more attempts
            for target in range(32, 36):
                if time.monotonic() > deadline - 5.0:
                    break
                for attempt in range(5000):
                    if time.monotonic() > deadline - 5.0:
                        break
                    # Use a wider range with some bias
                    if attempt % 3 == 0:
                        pool = list(range(1, 257))
                    elif attempt % 3 == 1:
                        pool = list(range(50, 257))
                    else:
                        pool = list(range(100, 257))
                    if len(pool) < target:
                        continue
                    cand = rng.sample(pool, target)
                    if _verify_fast(cand):
                        best = cand
                        best = _grow_greedy(best, deadline - 2.0)
                        break
                if len(best) >= target:
                    break
        
        # If still 31, try swap-based improvement on the base set
        if len(best) == 31 and time.monotonic() < deadline - 3.0:
            improved = _swap_improve(best, deadline - 3.0)
            if len(improved) > 31:
                best = improved
            # Try greedy growth again
            if time.monotonic() < deadline - 1.0:
                best = _grow_greedy(best, deadline - 1.0)
    
    # Final safety check
    if not _verify(best):
        best = list(range(226, 257))
    
    return best
