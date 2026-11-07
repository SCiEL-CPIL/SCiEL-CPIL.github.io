"""Find m in [0, 1000000] maximizing f(12, m) - the least L such that (m, m+L] has distinct b_i with i|b_i."""
import time
from collections import defaultdict, deque

# EVOLVE-BLOCK-START
N = 12
MAX_M = 1000000
MAX_L = 144  # f(12,m) is at most 144
PERIOD = 27720  # lcm(1..12)

def _hopcroft_karp(adj, n_left, n_right):
    """Bipartite matching. adj[u] = list of v. Returns matching size and matching dict."""
    match_l = [-1] * n_left
    match_r = [-1] * n_right
    dist = [0] * n_left
    
    def bfs():
        q = deque()
        found = False
        for u in range(n_left):
            if match_l[u] == -1:
                dist[u] = 0
                q.append(u)
            else:
                dist[u] = -1
        while q:
            u = q.popleft()
            for v in adj[u]:
                mu = match_r[v]
                if mu == -1:
                    found = True
                elif dist[mu] == -1:
                    dist[mu] = dist[u] + 1
                    q.append(mu)
        return found
    
    def dfs(u):
        for v in adj[u]:
            mu = match_r[v]
            if mu == -1 or (dist[mu] == dist[u] + 1 and dfs(mu)):
                match_l[u] = v
                match_r[v] = u
                return True
        dist[u] = -1
        return False
    
    matching = 0
    while bfs():
        for u in range(n_left):
            if match_l[u] == -1:
                if dfs(u):
                    matching += 1
    return matching

def _feasible(m, L):
    """Check if (m, m+L] contains 12 distinct b_i with i|b_i."""
    adj = []
    val_to_idx = {}
    
    for i in range(1, N + 1):
        first = ((m // i) + 1) * i
        if first <= m:
            first += i
        multiples = []
        v = first
        while v <= m + L:
            if v not in val_to_idx:
                val_to_idx[v] = len(val_to_idx)
            multiples.append(val_to_idx[v])
            v += i
        adj.append(multiples)
    
    n_right = len(val_to_idx)
    if n_right < N:
        return False
    
    matching = _hopcroft_karp(adj, N, n_right)
    return matching == N

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
