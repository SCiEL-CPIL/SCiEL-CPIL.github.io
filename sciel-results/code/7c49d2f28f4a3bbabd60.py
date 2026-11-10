"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

N = 12
LCM_12 = 27720  # LCM(1,2,...,12)
MAX_F = 144

def _has_perfect_matching(m, L):
    """Check if in (m, m+L] we can assign distinct b_i to each i in 1..12 with i|b_i."""
    adj = [None] * N
    for i in range(1, N + 1):
        r = (m + 1) % i
        first_p = (i - r) % i
        if first_p >= L:
            return False
        adj[i - 1] = list(range(first_p, L, i))
    
    # Order left nodes by candidate count (most constrained first)
    order = sorted(range(N), key=lambda i: len(adj[i]))
    
    match_r = [-1] * L
    match_r_size = L
    
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
    lo, hi = 12, MAX_F
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
    
    return int(best_m)
# EVOLVE-BLOCK-END
