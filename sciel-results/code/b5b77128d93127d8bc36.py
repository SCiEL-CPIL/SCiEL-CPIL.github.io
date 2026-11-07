import time
import sys
from functools import lru_cache

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s margin before 90s kill
    
    n = 12
    period = 27720  # LCM(1..12)
    
    def build_adj(L, m):
        """Build adjacency lists: adj[i-1] = list of positions j where (m+1+j) % i == 0"""
        adj = [[] for _ in range(n)]
        for i in range(1, n + 1):
            start_k = (m // i) + 1
            end_k = (m + L) // i
            for k in range(start_k, end_k + 1):
                val = k * i
                j = val - m - 1
                if 0 <= j < L:
                    adj[i - 1].append(j)
        return adj
    
    def has_matching_fast(adj, L):
        """Kuhn's algorithm with early termination via Hall's condition check on subsets of size 1 and 2."""
        # Quick Hall check: each adj[i] must be non-empty
        for i in range(n):
            if len(adj[i]) == 0:
                return False
        
        # Check Hall for pairs: |adj[i] ∪ adj[j]| >= 2 for all i != j
        for i in range(n):
            for j in range(i+1, n):
                union = set(adj[i]) | set(adj[j])
                if len(union) < 2:
                    return False
        
        # Full Kuhn's algorithm
        matchR = [-1] * L
        matchL = [-1] * n
        
        def dfs(u, seen):
            for v in adj[u]:
                if seen[v]:
                    continue
                seen[v] = True
                if matchR[v] == -1 or dfs(matchR[v], seen):
                    matchR[v] = u
                    matchL[u] = v
                    return True
            return False
        
        for u in range(n):
            seen = [False] * L
            if not dfs(u, seen):
                return False
        return True
    
    def check_matching(L, m):
        adj = build_adj(L, m)
        return has_matching_fast(adj, L)
    
    def f_of_m(m):
        """Compute f(12, m): least positive L with perfect matching using binary search."""
        # Lower bound: max over i of distance to next multiple of i
        min_L = 0
        for i in range(1, n + 1):
            next_mult = ((m // i) + 1) * i
            dist = next_mult - m
            if dist > min_L:
                min_L = dist
        
        lo = max(min_L, 1)
        hi = 144
        
        # Binary search for smallest L where matching exists
        # check_matching is monotone: if true for L, true for L+1
        while lo < hi:
            mid = (lo + hi) // 2
            if check_matching(mid, m):
                hi = mid
            else:
                lo = mid + 1
        
        if lo <= 144 and check_matching(lo, m):
            return lo
        return 144
    
    best_m = 0
    best_L = 0
    
    # Precompute all f(12, m) for m in [0, period-1]
    for m in range(period):
        if time.time() > deadline:
            break
        L_val = f_of_m(m)
        if L_val > best_L:
            best_L = L_val
            best_m = m
    
    return int(best_m)
