import time
import sys
from collections import deque

def _can_match(m, L, n=12):
    # Check if (m, m+L] contains distinct b_1..b_n with i | b_i
    # Use Hopcroft-Karp or simple DFS augmenting paths
    # For small n=12, DFS is fast enough
    # Build adjacency: for each i in 1..n, list of multiples of i in (m, m+L]
    adj = [[] for _ in range(n)]
    for i in range(1, n + 1):
        # smallest multiple of i > m
        r = ((m // i) + 1) * i
        end = m + L
        if r > end:
            return False
        v = r
        while v <= end:
            adj[i - 1].append(v)
            v += i
    
    # match_r: value -> left index (0-based), or None
    match_r = {}
    
    sys.setrecursionlimit(10000)
    
    def dfs(u, visited):
        for v in adj[u]:
            if v in visited:
                continue
            visited.add(v)
            if v not in match_r or dfs(match_r[v], visited):
                match_r[v] = u
                return True
        return False
    
    count = 0
    for u in range(n):
        visited = set()
        if dfs(u, visited):
            count += 1
        else:
            return False
    return count == n

def _f(m, n=12):
    # Binary search smallest L in [1, 144]
    lo, hi = 1, 144
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
    deadline = time.time() + 80  # 80s budget, 10s safety margin
    
    best_m = 12
    best_f = _f(12)
    
    # Strategy: 
    # 1. Coarse scan with step 7 to find good regions
    # 2. Refine around best found
    
    step = 7
    m = 0
    while m <= 1000000:
        if time.time() > deadline:
            break
        f = _f(m)
        if f > best_f:
            best_f = f
            best_m = m
        m += step
    
    # Refine around best_m with smaller step
    if time.time() < deadline:
        # First refine with step 1 in a wider range
        for dm in range(-500, 501):
            if time.time() > deadline:
                break
            mm = best_m + dm
            if mm < 0 or mm > 1000000:
                continue
            f = _f(mm)
            if f > best_f:
                best_f = f
                best_m = mm
    
    # Further refine with step 1 in a tighter range
    if time.time() < deadline:
        for dm in range(-50, 51):
            if time.time() > deadline:
                break
            mm = best_m + dm
            if mm < 0 or mm > 1000000:
                continue
            f = _f(mm)
            if f > best_f:
                best_f = f
                best_m = mm
    
    return int(best_m)
