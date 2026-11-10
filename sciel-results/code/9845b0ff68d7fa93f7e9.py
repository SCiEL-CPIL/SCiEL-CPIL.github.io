"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math

P = {'n': 12, 'maximum_start': 1000000}
LCM_12 = 27720  # LCM(1,2,...,12)

def _has_perfect_matching(m, L):
    """Check if in (m, m+L] we can assign distinct b_i to each i in 1..12 with i|b_i."""
    n = 12
    # Build adjacency: for each i, which positions in (m, m+L] are divisible by i
    adj = [[] for _ in range(n)]
    for i in range(1, n+1):
        first = m + 1
        rem = first % i
        if rem == 0:
            first_mult = first
        else:
            first_mult = first + (i - rem)
        for b in range(first_mult, m + L + 1, i):
            idx = b - m - 1
            if 0 <= idx < L:
                adj[i-1].append(idx)
    
    # Check if any left node has no candidates
    for i in range(n):
        if not adj[i]:
            return False
    
    # Order left nodes by candidate count (most constrained first)
    order = sorted(range(n), key=lambda i: len(adj[i]))
    
    # DFS-based bipartite matching
    match_r = [-1] * L
    match_l = [-1] * n
    
    def dfs(u, visited):
        for v in adj[u]:
            if visited[v]:
                continue
            visited[v] = True
            if match_r[v] == -1 or dfs(match_r[v], visited):
                match_r[v] = u
                match_l[u] = v
                return True
        return False
    
    for u in order:
        visited = [False] * L
        if not dfs(u, visited):
            return False
    return True

def _compute_f(m):
    """Compute f(12, m): least positive L such that (m, m+L] has distinct b_i with i|b_i."""
    lo, hi = 1, 144
    while lo < hi:
        mid = (lo + hi) // 2
        if _has_perfect_matching(m, mid):
            hi = mid
        else:
            lo = mid + 1
    return lo

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 85s deadline, 5s safety margin
    
    # Exploit periodicity: f(12, m) depends only on m mod LCM_12
    # So we only need to check m = 0, 1, ..., LCM_12 - 1
    best_m = 0
    best_f = 0
    
    # Exhaustive search over one full period
    for m in range(LCM_12):
        if time.time() > deadline - 5:
            break
        f_val = _compute_f(m)
        if f_val > best_f:
            best_f = f_val
            best_m = m
    
    # If we didn't finish the full period, do a local refinement around best_m
    if time.time() < deadline - 10:
        for dm in range(-50, 51):
            if time.time() > deadline - 5:
                break
            m2 = (best_m + dm) % LCM_12
            f_val = _compute_f(m2)
            if f_val > best_f:
                best_f = f_val
                best_m = m2
    
    # best_m is in [0, LCM_12), which is within [0, 1000000]
    return int(best_m)
# EVOLVE-BLOCK-END
