import time
import math
import random

def construct():
    start_time = time.time()
    limit = 85  # Leave 5s for overhead
    
    N = 12
    period = 27720
    
    def has_matching(m, L):
        """Check if there's a perfect matching for given m and L."""
        # Build candidate lists
        cands = [[] for _ in range(N + 1)]
        for i in range(1, N + 1):
            ri = (-m - 1) % i
            if ri < L:
                cands[i] = list(range(ri, L, i))
        
        # Quick fail: if any i has no candidates
        for i in range(1, N + 1):
            if not cands[i]:
                return False
        
        # Check if perfect matching exists using DFS (Kuhn)
        match_to = [-1] * L  # match_to[j] = i or -1
        
        def dfs(i, visited):
            for j in cands[i]:
                if visited[j]:
                    continue
                visited[j] = True
                if match_to[j] == -1 or dfs(match_to[j], visited):
                    match_to[j] = i
                    return True
            return False
        
        # Try to match all i from 1 to 12
        # Order: fewest candidates first
        order = list(range(1, N + 1))
        order.sort(key=lambda i: len(cands[i]))
        
        for i in order:
            visited = [False] * L
            if not dfs(i, visited):
                return False
        
        return True
    
    def compute_f(m):
        """Compute the least L >= 1 such that there's a perfect matching using binary search."""
        # Binary search on L in [1, 150]
        lo, hi = 1, 150
        best = 150
        while lo <= hi:
            mid = (lo + hi) // 2
            if has_matching(m, mid):
                best = mid
                hi = mid - 1
            else:
                lo = mid + 1
        return best
    
    best_m = 0
    best_f = 0
    
    # Phase 1: Random sampling to find a good initial candidate
    random.seed(42)
    sampled = set()
    num_random = 500
    for _ in range(num_random):
        if time.time() - start_time > limit - 30:
            break
        m = random.randint(0, period - 1)
        if m in sampled:
            continue
        sampled.add(m)
        f_val = compute_f(m)
        if f_val > best_f:
            best_f = f_val
            best_m = m
    
    # Phase 2: Local search around the best found so far
    if time.time() - start_time < limit - 20:
        for delta in range(-100, 101):
            if time.time() - start_time > limit - 20:
                break
            m_candidate = (best_m + delta) % period
            if m_candidate in sampled:
                continue
            sampled.add(m_candidate)
            f_val = compute_f(m_candidate)
            if f_val > best_f:
                best_f = f_val
                best_m = m_candidate
    
    # Phase 3: Full sweep if time permits
    if time.time() - start_time < limit - 5:
        for m in range(period):
            if time.time() - start_time > limit - 5:
                break
            if m in sampled:
                continue
            f_val = compute_f(m)
            if f_val > best_f:
                best_f = f_val
                best_m = m
    
    return best_m
