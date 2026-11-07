"""Find m in [0, 1000000] maximizing f(12, m) - the least L such that (m, m+L] has distinct b_i with i|b_i."""
import time
from collections import defaultdict, deque

# EVOLVE-BLOCK-START
N = 12
MAX_M = 1000000
MAX_L = 144  # f(12,m) is at most 144

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
    # Build adjacency: for each i (0..11 representing i+1), find multiples of (i+1) in (m, m+L]
    adj = []
    # Map from value to index for right side
    val_to_idx = {}
    right_vals = []
    
    for i in range(1, N + 1):
        # Multiples of i in (m, m+L]: first multiple > m, then step by i
        first = ((m // i) + 1) * i
        if first <= m:
            first += i
        multiples = []
        v = first
        while v <= m + L:
            if v not in val_to_idx:
                val_to_idx[v] = len(right_vals)
                right_vals.append(v)
            multiples.append(val_to_idx[v])
            v += i
        adj.append(multiples)
    
    n_right = len(right_vals)
    if n_right < N:
        return False
    
    matching = _hopcroft_karp(adj, N, n_right)
    return matching == N

def _compute_f(m):
    """Compute f(12, m) = least positive L such that (m, m+L] has valid distinct b_i."""
    # Binary search on L from 1 to MAX_L
    lo, hi = 1, MAX_L
    if not _feasible(m, MAX_L):
        return None  # shouldn't happen per problem statement
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
    
    best_m = 12
    best_f = 0
    
    # First, do a coarse scan to find promising regions
    # The key is that f(12,m) depends on the distribution of multiples
    # Let's scan m in steps and refine
    
    # Coarse scan: check every m in a smart way
    # Since MAX_L is small (144), the window is small
    # f(12, m) is periodic-ish in m with period lcm(1..12) = 27720
    
    # Let's scan m from 0 to MAX_M
    # But that's 1 million evaluations, each with binary search + matching
    # Need to be efficient
    
    # Strategy: scan m in blocks, use the fact that f changes slowly
    # First do a coarse pass with step=100, then refine around good spots
    
    # Coarse pass
    step = 50
    m = 0
    while m <= MAX_M:
        if time.time() > deadline:
            break
        f_val = _compute_f(m)
        if f_val is not None and f_val > best_f:
            best_f = f_val
            best_m = m
        m += step
    
    # Refine around the best region found
    if time.time() < deadline - 5:
        # Scan more finely around best_m
        refine_start = max(0, best_m - 200)
        refine_end = min(MAX_M, best_m + 200)
        for mm in range(refine_start, refine_end + 1):
            if time.time() > deadline:
                break
            f_val = _compute_f(mm)
            if f_val is not None and f_val > best_f:
                best_f = f_val
                best_m = mm
    
    # Also try some specific promising m values
    # m near multiples of large numbers might create gaps
    if time.time() < deadline - 5:
        # Try m values that are just below a multiple of 12, 11, etc.
        candidates = set()
        for i in range(1, N + 1):
            for k in range(0, MAX_M // i + 1):
                c = k * i - 1  # just before a multiple of i
                if 0 <= c <= MAX_M:
                    candidates.add(c)
                c2 = k * i  # at a multiple
                if 0 <= c2 <= MAX_M:
                    candidates.add(c2)
        
        # Limit candidates to keep time bounded
        candidate_list = sorted(candidates)
        # Sample if too many
        if len(candidate_list) > 5000:
            step_c = len(candidate_list) // 5000
            candidate_list = candidate_list[::step_c]
        
        for mm in candidate_list:
            if time.time() > deadline:
                break
            f_val = _compute_f(mm)
            if f_val is not None and f_val > best_f:
                best_f = f_val
                best_m = mm
    
    return int(best_m)
# EVOLVE-BLOCK-END
