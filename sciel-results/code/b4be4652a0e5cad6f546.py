"""Worst interval starts for distinct-multiple matching."""
import time

# EVOLVE-BLOCK-START

def _compute_f(m, n=12):
    """Compute f(n, m): least L>0 s.t. (m, m+L] has distinct b_i with i|b_i."""
    
    def can_match(L):
        adj = [[] for _ in range(n)]
        for i in range(1, n + 1):
            q = (m // i) + 1
            first = q * i
            if first <= m + L:
                idx = first - m - 1
                while idx < L:
                    adj[i - 1].append(idx)
                    idx += i
        
        # Process nodes with fewer candidates first
        order = sorted(range(n), key=lambda i: len(adj[i]))
        
        match = [-1] * L
        
        def dfs(i, visited):
            for j in adj[i]:
                if visited[j]:
                    continue
                visited[j] = True
                if match[j] == -1 or dfs(match[j], visited):
                    match[j] = i
                    return True
            return False
        
        found = 0
        for i in order:
            visited = [False] * L
            if dfs(i, visited):
                found += 1
        
        return found == n
    
    lo = 1
    hi = 144
    
    for i in range(1, n + 1):
        q = (m // i) + 1
        first = q * i
        lb = first - m
        if lb > lo:
            lo = lb
    
    if lo > hi:
        return hi
    
    while lo < hi:
        mid = (lo + hi) // 2
        if can_match(mid):
            hi = mid
        else:
            lo = mid + 1
    
    return lo

def construct():
    """Find m in [0, 1000000] that maximizes f(12, m)."""
    deadline = time.time() + 50
    
    best_m = 0
    best_f = 0
    
    period = 27720
    
    for m in range(period):
        if time.time() > deadline:
            break
        f_val = _compute_f(m)
        if f_val > best_f:
            best_f = f_val
            best_m = m
            
    return int(best_m)

# EVOLVE-BLOCK-END
