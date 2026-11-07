import time
import sys

def construct():
    """
    Finds an integer m in [0, 1000000] that maximizes f(12, m),
    where f(12, m) is the least positive L such that (m, m+L] contains
    distinct b_1...b_12 with i dividing b_i.
    """
    n = 12
    max_L = 144
    period = 27720  # lcm(1..12)
    
    start_time = time.time()
    time_limit = 85.0  # Safety margin before 90s kill
    
    best_m = 0
    best_L = 0
    
    def can_match(m, L):
        """
        Checks if there is a perfect matching for window (m, m+L].
        Returns True if valid distinct b_i exist.
        """
        if L < n:
            return False
        
        # Build adjacency: for each divisor i (1..12), which positions j in [0,L) work
        # Position j corresponds to value m+1+j
        # i divides m+1+j iff (m+1+j) % i == 0 iff j % i == (-(m+1)) % i
        adj = []
        for i in range(1, n + 1):
            target = (-(m + 1)) % i
            mask = 0
            j = target
            if j < L:
                while j < L:
                    mask |= (1 << j)
                    j += i
            adj.append(mask)
        
        # Sort divisors by number of available positions (most constrained first)
        # We need to track which divisor corresponds to which sorted position
        # For existence check, we can just sort the masks
        adj.sort(key=lambda x: x.bit_count())
        
        # If any divisor has no available positions, fail immediately
        if adj[0] == 0:
            return False
        
        # DFS with memoization
        # State: (index in sorted adj, used_mask)
        # used_mask is a bitmask of positions already assigned
        memo = {}
        
        def dfs(k, used_mask):
            if k == n:
                return True
            
            key = (k, used_mask)
            if key in memo:
                return memo[key]
            
            avail = adj[k] & (~used_mask)
            # Try each available position
            while avail:
                lsb = avail & (-avail)
                avail ^= lsb
                if dfs(k + 1, used_mask | lsb):
                    memo[key] = True
                    return True
            
            memo[key] = False
            return False
        
        return dfs(0, 0)
    
    # Search over one period
    for m in range(period):
        if time.time() - start_time > time_limit:
            break
        
        # Binary search for the minimum L that works
        # We know L is between n (12) and max_L (144)
        # If L works, any larger L also works (superset of positions)
        lo = n
        hi = max_L
        min_L = max_L + 1
        
        while lo <= hi:
            mid = (lo + hi) // 2
            if can_match(m, mid):
                min_L = mid
                hi = mid - 1
            else:
                lo = mid + 1
        
        if min_L < max_L + 1:
            # Found a valid L for this m
            if min_L > best_L:
                best_L = min_L
                best_m = m
        else:
            # No valid L up to max_L, which shouldn't happen by problem definition
            # But if it does, we can't use this m
            pass
    
    return int(best_m)
