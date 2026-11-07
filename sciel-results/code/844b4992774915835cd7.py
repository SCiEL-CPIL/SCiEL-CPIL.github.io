import time
import math

def construct():
    """
    Finds m in [0, 1000000] maximizing f(12, m).
    f(12, m) is the least L > 0 such that (m, m+L] contains distinct b_1..b_12 with i|b_i.
    """
    N = 12
    PERIOD = 27720  # LCM(1..12)
    
    def has_matching(r, L):
        # r is the residue, interval is (r, r+L]
        # Left nodes: 1..N
        # Right nodes: indices 0..L-1 corresponding to values r+1 .. r+L
        
        match_to = [-1] * L
        
        candidates = [[] for _ in range(N + 1)]
        
        for i in range(1, N + 1):
            rem = r % i
            if rem == 0:
                first = r + i
            else:
                first = r + (i - rem)
            
            for val in range(first, r + L + 1, i):
                j = val - (r + 1)
                candidates[i].append(j)
        
        # Sort left nodes by number of candidates (fewest first) for better pruning
        order = sorted(range(1, N + 1), key=lambda i: len(candidates[i]))
        
        def dfs(i, visited):
            for j in candidates[i]:
                if not visited[j]:
                    visited[j] = True
                    if match_to[j] == -1 or dfs(match_to[j], visited):
                        match_to[j] = i
                        return True
            return False
        
        for i in order:
            visited = [False] * L
            if not dfs(i, visited):
                return False
        return True
    
    best_m = 0
    max_L = 0
    
    start_time = time.time()
    deadline = start_time + 85.0
    
    for r in range(PERIOD):
        if time.time() > deadline:
            break
            
        # Binary search for minimum L
        lo, hi = 12, 144
        # Find minimum L in [lo, hi] where matching exists
        # Predicate: has_matching(r, L) is monotone non-decreasing in L
        while lo < hi:
            mid = (lo + hi) // 2
            if has_matching(r, mid):
                hi = mid
            else:
                lo = mid + 1
        
        if lo <= 144:
            L_min = lo
            if L_min > max_L:
                max_L = L_min
                best_m = r
                if max_L == 144:
                    return best_m
        # If lo > 144, no matching found in range, but problem says max is 144
    
    return best_m
