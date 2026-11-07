"""Find m in [0, 1000000] maximizing f(12, m) - the least L such that (m, m+L] has distinct b_i with i|b_i."""
import time
from collections import defaultdict, deque

# EVOLVE-BLOCK-START
N = 12
MAX_M = 1000000
MAX_L = 144  # f(12,m) is at most 144
PERIOD = 27720  # lcm(1..12)


def _feasible(m, L):
    """Check if (m, m+L] contains 12 distinct b_i with i|b_i."""
    # The integers in (m, m+L] are m+1, ..., m+L.
    # Map value v to index v - (m+1), which is in [0, L-1].
    adj = []
    for i in range(1, N + 1):
        first = ((m // i) + 1) * i
        if first <= m:
            first += i
        multiples = []
        v = first
        while v <= m + L:
            idx = v - (m + 1)
            multiples.append(idx)
            v += i
        if not multiples:
            return False
        adj.append(multiples)
    
    # Use a simple DFS for bipartite matching (Kuhn's algorithm)
    # This is often faster for small graphs than Hopcroft-Karp due to lower overhead.
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
    
    # Process nodes with fewer options first to prune early
    # adj[u] corresponds to i = u+1. Smaller i -> more options.
    # So u=N-1 (i=12) has fewest options.
    for u in range(N - 1, -1, -1):
        visited = [False] * L
        if not dfs(u, visited):
            return False
            
    return True

def _compute_f(m):
    """Compute f(12, m) = least positive L such that (m, m+L] has valid distinct b_i."""
    lo, hi = 1, MAX_L
    if not _feasible(m, MAX_L):
        return None
    while lo < hi:
        mid = (lo + hi) // 2
        if _feasible(m, mid):
            hi = mid
        else:
            lo = mid + 1
    return lo

def construct():
    start_time = time.time()
    deadline = start_time + 80  # 80s safety margin for 90s limit
    
    # f(12, m) depends only on m mod PERIOD
    # So we only need to evaluate PERIOD distinct residues
    # Then pick the best residue r and find the smallest m in [0, MAX_M] with m % PERIOD == r
    
    best_f = 0
    best_residue = 0
    
    # Evaluate all residues 0..PERIOD-1
    for r in range(PERIOD):
        if time.time() > deadline:
            break
        f_val = _compute_f(r)
        if f_val is not None and f_val > best_f:
            best_f = f_val
            best_residue = r
    
    # Find the smallest m in [0, MAX_M] with m % PERIOD == best_residue
    # m = best_residue (if best_residue <= MAX_M)
    # or m = best_residue + k*PERIOD for some k
    best_m = best_residue
    if best_m > MAX_M:
        # This shouldn't happen since PERIOD=27720 < MAX_M=1000000
        best_m = best_residue % (MAX_M + 1)
    
    # Verify and also check nearby values for robustness
    # Do a local search around best_m to confirm
    if time.time() < deadline - 5:
        search_start = max(0, best_m - 5)
        search_end = min(MAX_M, best_m + 5)
        for mm in range(search_start, search_end + 1):
            if time.time() > deadline:
                break
            f_val = _compute_f(mm)
            if f_val is not None and f_val > best_f:
                best_f = f_val
                best_m = mm
    
    return int(best_m)
# EVOLVE-BLOCK-END
