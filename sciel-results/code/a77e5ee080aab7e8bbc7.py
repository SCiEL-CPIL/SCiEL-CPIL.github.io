import time
import sys

def _can_match(m, L, n=12):
    """
    Check if (m, m+L] contains distinct b_1...b_n with i | b_i.
    Uses bitmask DFS with most-constrained-first ordering.
    """
    options = []
    for i in range(1, n + 1):
        mask = 0
        target = (-(m + 1)) % i
        for j in range(L):
            if j % i == target:
                mask |= (1 << j)
        if mask == 0:
            return False
        options.append((mask.bit_count(), mask))
    
    options.sort(key=lambda x: x[0])
    
    n_opts = n
    
    def dfs(k, used_mask):
        if k == n_opts:
            return True
        mask = options[k][1]
        available = mask & ~used_mask
        while available:
            bit = available & (-available)
            available -= bit
            if dfs(k + 1, used_mask | bit):
                return True
        return False
    
    return dfs(0, 0)

def _f_val(m, n=12):
    """
    Find the least L such that (m, m+L] allows a matching.
    Binary search on monotone property.
    """
    lo = n
    hi = 144
    
    if not _can_match(m, hi, n):
        return hi
    
    while lo < hi:
        mid = (lo + hi) // 2
        if _can_match(m, mid, n):
            hi = mid
        else:
            lo = mid + 1
    
    return lo

def construct():
    start_time = time.time()
    deadline = start_time + 80
    best_m = 0
    best_L = 0
    
    period = 27720
    
    for m in range(period):
        if time.time() > deadline:
            break
        
        L = _f_val(m)
        if L > best_L:
            best_L = L
            best_m = m
    
    return best_m
