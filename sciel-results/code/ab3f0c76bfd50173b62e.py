import time
import sys
from functools import lru_cache

def _can_match(m, L, n=12):
    """
    Check if (m, m+L] contains distinct b_1...b_n with i | b_i.
    Uses optimized DFS backtracking with memoization on used mask.
    """
    # Precompute valid indices for each i
    options = []
    for i in range(1, n + 1):
        # We need j in [0, L-1] such that (m + 1 + j) % i == 0
        # j % i == (-(m + 1)) % i
        target_mod = (-(m + 1)) % i
        opts = []
        # Find first j >= 0 such that j % i == target_mod
        first_j = target_mod
        if first_j >= L:
            # No options
            return False
        for j in range(first_j, L, i):
            opts.append(j)
        if not opts:
            return False
        options.append((len(opts), i, opts))
    
    # Sort by number of options (most constrained first)
    options.sort(key=lambda x: x[0])
    
    # If L is small enough, we can use bitmask for used
    if L <= 20:
        used_mask = 0
        opts_lists = [opts for _, _, opts in options]
        
        @lru_cache(maxsize=None)
        def dfs(k, mask):
            if k == n:
                return True
            for idx in opts_lists[k]:
                bit = 1 << idx
                if not (mask & bit):
                    if dfs(k + 1, mask | bit):
                        return True
            return False
        
        return dfs(0, 0)
    else:
        # Use array-based approach for larger L
        used = [False] * L
        opts_lists = [opts for _, _, opts in options]
        
        def dfs(k):
            if k == n:
                return True
            for idx in opts_lists[k]:
                if not used[idx]:
                    used[idx] = True
                    if dfs(k + 1):
                        return True
                    used[idx] = False
            return False
        
        return dfs(0)

def _f_val(m, n=12):
    """
    Find the least L such that (m, m+L] allows a matching.
    Binary search L in [1, 144].
    """
    low = 1
    high = 144
    best_L = 144 # Fallback
    
    while low <= high:
        mid = (low + high) // 2
        if _can_match(m, mid, n):
            best_L = mid
            high = mid - 1
        else:
            low = mid + 1
            
    return best_L

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin before 90s
    best_m = 0
    best_L = 0
    period = 27720 # lcm(1..12)
    
    # We only need to check one period
    for m in range(period):
        # Time guard
        if time.time() > deadline:
            break
        
        L = _f_val(m)
        if L > best_L:
            best_L = L
            best_m = m
            
    return best_m
