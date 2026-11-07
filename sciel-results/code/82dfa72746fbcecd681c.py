import time

def _is_valid(arr):
    """Check: distinct ints in 1..256, size 1..60, cardinality-separated subset sums."""
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
    # dp[k] is a bitset: bit j set means sum j is achievable with exactly k elements
    max_sum = sum(arr)
    dp = [0] * (n + 1)
    dp[0] = 1  # empty subset, sum 0
    for x in arr:
        for k in range(n, 0, -1):
            dp[k] |= (dp[k - 1] << x)
    # Check no sum appears in two different cardinalities
    seen = 0
    for k in range(n + 1):
        if dp[k] & seen:
            return False
        seen |= dp[k]
    return True

def _try_add(arr, candidate, n):
    """Check if adding candidate to arr preserves validity (incremental check)."""
    # Quick check: candidate not already in arr
    if candidate in arr:
        return False
    # We need to verify the full constraint after adding.
    # Efficient approach: compute new dp and check overlap.
    new_n = n + 1
    max_sum = sum(arr) + candidate
    dp = [0] * (new_n + 1)
    dp[0] = 1
    for x in arr:
        for k in range(new_n, 0, -1):
            dp[k] |= (dp[k - 1] << x)
    # Add candidate
    for k in range(new_n, 0, -1):
        dp[k] |= (dp[k - 1] << candidate)
    seen = 0
    for k in range(new_n + 1):
        if dp[k] & seen:
            return False
        seen |= dp[k]
    return True

def _grow(start, deadline):
    """Greedy growth: try to add elements one by one."""
    cur = list(start)
    cur_set = set(cur)
    # Sort candidates: prefer small values first (they're less likely to cause collisions)
    all_candidates = list(range(1, 257))
    # Remove already-present
    candidates = [c for c in all_candidates if c not in cur_set]
    
    # Try adding in order of smallest value first
    while time.monotonic() < deadline:
        added = False
        for c in candidates[:]:
            if c in cur_set:
                candidates.remove(c)
                continue
            if _try_add(cur, c, len(cur)):
                cur.append(c)
                cur_set.add(c)
                candidates.remove(c)
                added = True
                break
        if not added:
            break
    return cur

def construct():
    start_time = time.monotonic()
    deadline = start_time + 85.0  # safety margin
    
    # Start with a small valid set and grow
    # Try different starting sets
    best = []
    
    # Strategy: start with {1} and greedily add
    # But 1 might be too small. Let's try various starts.
    
    # Actually, let's think about what makes a good set.
    # If all elements are > sum of all smaller elements, then all subset sums are distinct (superincreasing).
    # But we want cardinality separation, not full distinctness.
    # 
    # Key insight: if element x > sum of all previously chosen elements, then any subset containing x
    # has sum > any subset not containing x. This gives strong separation.
    # But we only need cardinality separation.
    #
    # Let's try: start with 1, then each next element > sum of all previous.
    # 1, 2, 4, 8, ... powers of 2 up to 128. That's 8 elements (1,2,4,8,16,32,64,128).
    # 128+64=192, next would be 192 but max is 256. 192 works? 192 > 128+64+...+1 = 255? No, 192 < 255.
    # So superincreasing stops at 128. That gives 8 elements.
    #
    # Better approach: use the greedy with DP verification.
    
    # Start with empty-ish, try to build up
    # Try starting with just {1}
    cur = [1]
    best = _grow(cur, deadline)
    
    if len(best) > len(best):
        pass  # placeholder
    
    # Try different starting points
    for start_val in [1, 2, 3, 5, 10]:
        if time.monotonic() > deadline - 5:
            break
        cur = [start_val]
        result = _grow(cur, deadline)
        if len(result) > len(best):
            best = result
    
    # Try starting with a small set of well-chosen elements
    # e.g., powers of 2 up to some point
    if time.monotonic() < deadline - 10:
        powers = [1, 2, 4, 8, 16, 32, 64, 128]
        # Verify it's valid
        if _is_valid(powers):
            result = _grow(powers, deadline)
            if len(result) > len(best):
                best = result
    
    # Try a different strategy: start with moderately large numbers
    # to leave room for many small numbers
    if time.monotonic() < deadline - 10:
        # Start with {100, 200} and try to add small numbers
        for start_set in [[50, 100, 150], [40, 80, 120, 160], [30, 60, 90, 120, 150, 180, 210, 240]]:
            if not _is_valid(start_set):
                continue
            result = _grow(start_set, deadline)
            if len(result) > len(best):
                best = result
    
    return best
# EVOLVE-BLOCK-END
