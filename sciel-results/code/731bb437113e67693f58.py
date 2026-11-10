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
        
        # Process nodes with fewer candidates first for faster matching
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
    deadline = time.time() + 55
    
    best_m = 0
    best_f = 0
    
    period = 27720
    
    # First pass: scan one full period to find the best residue
    for m in range(period):
        if time.time() > deadline - 10:
            break
        f_val = _compute_f(m)
        if f_val > best_f:
            best_f = f_val
            best_m = m
    
    # Second pass: local refinement around the best candidates found
    if time.time() < deadline - 5:
        # Explore neighbors of the best m in both directions
        for delta in range(-20, 21):
            if time.time() > deadline - 2:
                break
            m_candidate = best_m + delta
            if 0 <= m_candidate < period:
                f_val = _compute_f(m_candidate)
                if f_val > best_f:
                    best_f = f_val
                    best_m = m_candidate
    
    # Third pass: if we have more time, check a few other high-value residues
    if time.time() < deadline - 2:
        # Check residues that are offsets from best_m by small amounts
        for offset in [1, -1, 2, -2, 3, -3, 6, -6, 12, -12]:
            if time.time() > deadline - 1:
                break
            m_candidate = (best_m + offset) % period
            if 0 <= m_candidate <= 1000000:
                f_val = _compute_f(m_candidate)
                if f_val > best_f:
                    best_f = f_val
                    best_m = m_candidate
    
    return int(best_m)

# EVOLVE-BLOCK-END
