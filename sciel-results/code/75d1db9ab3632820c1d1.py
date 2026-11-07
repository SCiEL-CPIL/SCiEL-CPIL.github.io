import time
import math
from functools import reduce
from collections import deque

def _lcm(a, b):
    return a * b // math.gcd(a, b)

def _hopcroft_karp(adj, n_left, n_right):
    pair_u = [-1] * n_left
    pair_v = [-1] * n_right
    dist = [0] * n_left
    
    def bfs():
        q = deque()
        for u in range(n_left):
            if pair_u[u] == -1:
                dist[u] = 0
                q.append(u)
            else:
                dist[u] = -1
        found = False
        while q:
            u = q.popleft()
            for v in adj[u]:
                pu = pair_v[v]
                if pu == -1:
                    found = True
                elif dist[pu] == -1:
                    dist[pu] = dist[u] + 1
                    q.append(pu)
        return found

    def dfs(u):
        for v in adj[u]:
            pu = pair_v[v]
            if pu == -1 or (dist[pu] == dist[u] + 1 and dfs(pu)):
                pair_u[u] = v
                pair_v[v] = u
                return True
        dist[u] = -1
        return False

    matching = 0
    while bfs():
        for u in range(n_left):
            if pair_u[u] == -1:
                if dfs(u):
                    matching += 1
    return matching

def _can_match(m, L, n=12):
    if L < n:
        return False
        
    val_to_idx = {}
    right_vals = []
    adj_idx = [[] for _ in range(n)]
    
    for i in range(n):
        idx = i + 1
        start_k = m // idx + 1
        end_k = (m + L) // idx
        
        if start_k > end_k:
            return False
            
        for k in range(start_k, end_k + 1):
            v = k * idx
            if v not in val_to_idx:
                val_to_idx[v] = len(right_vals)
                right_vals.append(v)
            adj_idx[i].append(val_to_idx[v])
            
    if len(right_vals) < n:
        return False
        
    return _hopcroft_karp(adj_idx, n, len(right_vals)) == n

def _f(m, n=12):
    # Binary search smallest L in [12, 144]
    # L=144 is always feasible
    lo, hi = 12, 143
    if not _can_match(m, hi, n):
        return 144
        
    while lo < hi:
        mid = (lo + hi) // 2
        if _can_match(m, mid, n):
            hi = mid
        else:
            lo = mid + 1
    return lo

def construct():
    start_time = time.time()
    deadline = start_time + 80  # 10s safety margin
    
    # Calculate LCM of 1..12
    p = reduce(_lcm, range(1, 13))
    
    # Check all m in [0, p-1] since f(12, m) is periodic with period p
    best_m = 0
    best_f = _f(0)
    
    for m in range(1, p):
        if time.time() > deadline:
            break
        f_val = _f(m)
        if f_val > best_f:
            best_f = f_val
            best_m = m

    # The best m in [0, p-1] is equivalent to any m + k*p.
    # We return the smallest such m.
    return int(best_m)
