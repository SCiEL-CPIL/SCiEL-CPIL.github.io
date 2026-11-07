import time

def _can_match(m, L, n=12):
    """
    Check if (m, m+L] contains distinct b_1..b_n with i | b_i.
    Uses bitmask and DFS.
    """
    # options[i] will be a bitmask of positions j in [0, L) such that (m+1+j) % i == 0
    options = []
    
    # Pre-calculate base offset
    base = m + 1
    
    for i in range(1, n + 1):
        # Find first j in [0, L) such that (base + j) % i == 0
        # base + j = k*i  => j = k*i - base
        # We need j >= 0 => k*i >= base => k >= ceil(base/i)
        # And j < L => k*i - base < L => k*i < base + L => k < (base+L)/i
        
        # Alternatively:
        # (base + j) % i == 0  <=> j % i == (-base) % i
        target = (-base) % i
        
        mask = 0
        # Iterate through valid j
        # j = target, target + i, target + 2i, ...
        # Ensure target < L
        if target < L:
            # Set bits
            # Unrolling or loop
            for j in range(target, L, i):
                mask |= (1 << j)
        
        if mask == 0:
            return False
        
        options.append((mask.bit_count(), mask))
        
    # Sort by number of available positions (most constrained first)
    options.sort(key=lambda x: x[0])
    
    n_opts = n
    
    # DFS
    # We use a list of masks. 
    # To speed up, we can use a non-recursive approach or just standard recursion with bitmasks.
    # Given n=12, recursion depth is small.
    
    def dfs(k, used_mask):
        if k == n_opts:
            return True
        
        count, mask = options[k]
        avail = mask & ~used_mask
        
        # If no available, fail
        if avail == 0:
            return False
        
        # Heuristic: Try the position that is most "constrained" for future steps?
        # Or just iterate.
        # To improve pruning, we can check if remaining sets have enough capacity.
        # But for n=12, simple iteration is often fast enough.
        
        while avail:
            # Get lowest set bit
            low = avail & (-avail)
            avail ^= low
            
            if dfs(k + 1, used_mask | low):
                return True
                
        return False

    return dfs(0, 0)

def _f_val(m, n=12):
    """
    Compute f(n, m): least positive L such that (m, m+L] has distinct b_i with i|b_i.
    """
    # L must be at least n.
    # Upper bound is small.
    lo = n
    hi = 200  # Safe upper bound
    
    # Binary search
    # First check if lo works
    if not _can_match(m, lo, n):
        while lo < hi:
            mid = (lo + hi) // 2
            if _can_match(m, mid, n):
                hi = mid
            else:
                lo = mid + 1
        return lo
    else:
        return lo

def construct():
    start_time = time.time()
    deadline = 85.0  # Safety margin
    
    best_m = 0
    best_L = 0
    
    # The function f(12, m) is periodic with period 27720
    period = 27720
    
    # We only need to check m in [0, period)
    # Then we can return any m in [0, 1000000] that is congruent to best_m mod period.
    # The problem asks for m in [0, 1000000].
    # So if best_m is found in [0, 27720), we can just return best_m (since 27720 < 1000000).
    
    for m in range(period):
        if time.time() - start_time > deadline:
            break
        
        L = _f_val(m)
        
        if L > best_L:
            best_L = L
            best_m = m
            
    # best_m is in [0, 27720), which is within [0, 1000000]
    return int(best_m)
