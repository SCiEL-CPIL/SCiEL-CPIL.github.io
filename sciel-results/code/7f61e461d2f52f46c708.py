"""Search for m maximizing f(12, m) within the 90s wall-clock limit."""
import time
from collections import deque

def _can_match(m, L, n=12):
    """Check if (m, m+L] contains distinct b_1..b_n with i | b_i, using Hopcroft-Karp."""
    if L < n:
        return False
    # nums in (m, m+L]
    # For each i in 1..n, the candidate numbers are multiples of i in (m, m+L]
    # Build adjacency: left node i (0-indexed) -> right nodes (actual numbers)
    # Right nodes are the L numbers m+1 .. m+L
    # To keep matching fast, index right nodes 0..L-1
    adj = [[] for _ in range(n)]
    for i in range(1, n + 1):
        # first multiple of i in (m, m+L]: smallest x > m with x % i == 0
        first = (m // i + 1) * i
        # iterate multiples
        for x in range(first, m + L + 1, i):
            idx = x - (m + 1)  # 0-indexed in the interval
            if 0 <= idx < L:
                adj[i - 1].append(idx)
            else:
                break
        if not adj[i - 1]:
            return False
    # Hopcroft-Karp
    INF = float('inf')
    pair_u = [-1] * n
    pair_v = [-1] * L
    dist = [0] * n

    def bfs():
        dq = deque()
        for u in range(n):
            if pair_u[u] == -1:
                dist[u] = 0
                dq.append(u)
            else:
                dist[u] = INF
        found = False
        while dq:
            u = dq.popleft()
            for v in adj[u]:
                pu = pair_v[v]
                if pu == -1:
                    found = True
                elif dist[pu] == INF:
                    dist[pu] = dist[u] + 1
                    dq.append(pu)
        return found

    def dfs(u):
        for v in adj[u]:
            pu = pair_v[v]
            if pu == -1 or (dist[pu] == dist[u] + 1 and dfs(pu)):
                pair_u[u] = v
                pair_v[v] = u
                return True
        dist[u] = INF
        return False

    matching = 0
    while bfs():
        for u in range(n):
            if pair_u[u] == -1:
                if dfs(u):
                    matching += 1
        if matching == n:
            return True
    return matching == n

def _f_val(m, n=12, Lmax=144):
    """Compute f(n, m): least positive L such that (m, m+L] has distinct b_i with i|b_i."""
    # lower bound: need at least n numbers
    lo = n
    hi = Lmax
    if not _can_match(m, hi, n):
        # shouldn't happen for reasonable m, but return hi as fallback
        return hi
    while lo < hi:
        mid = (lo + hi) // 2
        if _can_match(m, mid, n):
            hi = mid
        else:
            lo = mid + 1
    return lo

def construct():
    start = time.time()
    deadline = start + 80.0  # safety margin before 90s
    best_m = 12
    best_f = _f_val(12)
    
    # Phase 1: coarse scan of m values
    # Try a spread of m values across the range
    candidates = []
    # Include small values, and spread across the range
    for m in range(0, 200):
        candidates.append(m)
    # Add spaced samples across the full range
    step = 500
    for m in range(0, 1000001, step):
        candidates.append(m)
    
    # Evaluate candidates, tracking best
    for m in candidates:
        if time.time() > deadline - 10.0:
            break
        f = _f_val(m)
        if f > best_f:
            best_f = f
            best_m = m
    
    # Phase 2: local refinement around the best m
    if time.time() < deadline - 5.0:
        # Search a window around best_m
        window = 200
        for m in range(max(0, best_m - window), min(1000000, best_m + window) + 1):
            if time.time() > deadline - 2.0:
                break
            if m == best_m:
                continue
            f = _f_val(m)
            if f > best_f:
                best_f = f
                best_m = m
    
    # Phase 3: if we still have time, do a broader secondary scan
    if time.time() < deadline - 5.0:
        # Try some structured patterns: multiples of LCM(1..12)=27720
        lcm_12 = 27720
        for offset in range(0, lcm_12):
            if time.time() > deadline - 2.0:
                break
            for base in range(0, 1000001, lcm_12):
                m = base + offset
                if m > 1000000:
                    break
                if m == best_m:
                    continue
                f = _f_val(m)
                if f > best_f:
                    best_f = f
                    best_m = m
    
    return int(best_m)
