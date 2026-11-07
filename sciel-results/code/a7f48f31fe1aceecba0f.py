import time

def _can_match(m, L, n=12):
    # Precompute masks for each i in 1..n
    # mask[i] has bit j set if (m+1+j) % i == 0
    masks = [0] * (n + 1)
    for i in range(1, n + 1):
        target = (-(m + 1)) % i
        mask = 0
        for j in range(L):
            if j % i == target:
                mask |= (1 << j)
        masks[i] = mask
        
    # If any mask is 0, impossible
    for i in range(1, n + 1):
        if masks[i] == 0:
            return False

    # DFS with MRV (Minimum Remaining Values)
    # We maintain a list of unassigned indices
    # To speed up, we sort indices by popcount of their mask initially?
    # Actually, dynamic MRV is better.
    
    # Since n=12, we can just use a simple recursive solver with bitsets.
    # State: used_mask (bits of positions already taken)
    # We need to assign indices 1..n.
    
    # Let's try a different approach: Hopcroft-Karp is overkill. 
    # Simple DFS with ordering by smallest mask size is good.
    
    # However, calculating popcount every step is expensive in Python.
    # Let's pre-sort indices by the number of options they have for this specific m, L.
    # The options for i are masks[i] & ~used_mask. The size changes as we assign.
    
    # Alternative: Bitmask DP? 
    # State: subset of indices assigned? No, we need to track which positions are used.
    # State: used_positions (L bits). We need to check if we can assign all 12 indices.
    # This is equivalent to: does there exist a matching?
    
    # Let's stick to DFS but optimize the inner loop.
    
    indices = list(range(1, n + 1))
    
    # Sort indices by initial popcount to prune early
    # But popcount changes. 
    # Let's just use a fixed order 1..n and rely on the small n.
    # Or, better, sort by popcount(masks[i]) once.
    indices.sort(key=lambda i: masks[i].bit_count())
    
    def dfs(k, used_mask):
        if k == n:
            return True
        i = indices[k]
        available = masks[i] & ~used_mask
        while available:
            # Get least significant bit
            bit = available & (-available)
            if dfs(k + 1, used_mask | bit):
                return True
            available ^= bit
        return False
    
    return dfs(0, 0)

def _f_val(m, n=12):
    # Binary search for min L
    # Lower bound: n (need at least n distinct numbers)
    # Upper bound: 144 (from task description)
    lo = n
    hi = 144
    
    if not _can_match(m, hi, n):
        return hi # Should not happen usually, but safe
    
    while lo < hi:
        mid = (lo + hi) // 2
        if _can_match(m, mid, n):
            hi = mid
        else:
            lo = mid + 1
            
    return lo

def construct():
    start_time = time.time()
    deadline = start_time + 85 # Safety margin for 90s limit
    
    best_m = 0
    best_L = 0
    
    period = 27720 # LCM(1..12)
    
    # We scan m in [0, period). 
    # If time runs out, we return best found.
    for m in range(period):
        if time.time() > deadline:
            break
            
        L = _f_val(m)
        if L > best_L:
            best_L = L
            best_m = m
            
    return best_m
