"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

P = {'n': 12, 'maximum_start': 1000000}
LCM_12 = 27720  # LCM(1,2,...,12)

def _has_perfect_matching(m, L):
    """Check if in (m, m+L] we can assign distinct b_i to each i in 1..12 with i|b_i."""
    n = 12
    # Build adjacency: for each i, which positions in (m, m+L] are divisible by i
    adj = [None] * n
    for i in range(1, n+1):
        first = m + 1
        rem = first % i
        if rem == 0:
            first_mult = first
        else:
            first_mult = first + (i - rem)
        start_pos = first_mult - m - 1
        if start_pos >= L:
            return False
        adj[i-1] = list(range(start_pos, L, i))
    
    # Order left nodes by candidate count (most constrained first)
    order = sorted(range(n), key=lambda i: len(adj[i]))
    
    # DFS-based bipartite matching
    match_r = [-1] * L
    
    def dfs(u, visited):
        for v in adj[u]:
            if visited[v]:
                continue
            visited[v] = True
            if match_r[v] == -1 or dfs(match_r[v], visited):
                match_r[v] = u
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
    
    # best_m is in [0, LCM_12), which is within [0, 1000000]
    return int(best_m)
# EVOLVE-BLOCK-END
