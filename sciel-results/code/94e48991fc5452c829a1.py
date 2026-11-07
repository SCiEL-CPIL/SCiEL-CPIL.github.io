"""Rewritten program: search for m maximizing f(12, m) via bipartite matching."""
# EVOLVE-BLOCK-START
import time
import math
from collections import deque

N = 12
MAX_M = 1000000
LCM_12 = 27720  # lcm(1..12)

def _hopcroft_karp(adj, n_left, n_right):
    """Exact bipartite matching. adj: list of lists (neighbors of each left node)."""
    pair_u = [-1] * n_left
    pair_v = [-1] * n_right
    dist = [0] * n_left

    def bfs():
        q = deque()
        found = False
        for u in range(n_left):
            if pair_u[u] == -1:
                dist[u] = 0
                q.append(u)
            else:
                dist[u] = -1
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
            if pair_u[u] == -1 and dfs(u):
                matching += 1
    return matching

def _has_matching(m, L, multiples):
    """Check if (m, m+L] contains distinct b_i with i|b_i."""
    lo = m + 1
    hi = m + L
    adj = []
    for i in range(1, N + 1):
        mults = multiples[i]
        # find multiples in [lo, hi]
        # mults is sorted list of multiples of i in [0, MAX_M+200]
        # binary search for first >= lo
        import bisect
        idx = bisect.bisect_left(mults, lo)
        neighbors = []
        while idx < len(mults) and mults[idx] <= hi:
            neighbors.append(mults[idx])
            idx += 1
        if not neighbors:
            return False
        adj.append(neighbors)
    # Hopcroft-Karp
    return _hopcroft_karp(adj, N, len(set(x for a in adj for x in a))) >= N

def _f_value(m, multiples, max_L=200):
    """Compute f(12, m): least positive L with valid matching, or None if > max_L."""
    # Binary search on L in [1, max_L]
    lo, hi = 1, max_L
    best = None
    while lo <= hi:
        mid = (lo + hi) // 2
        if _has_matching(m, mid, multiples):
            best = mid
            hi = mid - 1
        else:
            lo = mid + 1
    return best

def construct():
    start_time = time.time()
    deadline = start_time + 80  # 80s safety margin before 90s kill

    # Precompute multiples for each i in [0, MAX_M + 200]
    max_val = MAX_M + 200
    multiples = {}
    for i in range(1, N + 1):
        multiples[i] = list(range(0, max_val + 1, i))

    # Candidate m values: near multiples of 8, 9, and LCM boundaries
    candidates = set()

    # Near multiples of 8 (tightest constraint for even positions)
    for k in range(0, max_val // 8 + 1):
        m = k * 8 - 1  # just before a multiple of 8
        if 0 <= m <= MAX_M:
            candidates.add(m)
        m2 = k * 8  # at a multiple of 8
        if 0 <= m2 <= MAX_M:
            candidates.add(m2)

    # Near multiples of 9
    for k in range(0, max_val // 9 + 1):
        m = k * 9 - 1
        if 0 <= m <= MAX_M:
            candidates.add(m)
        m2 = k * 9
        if 0 <= m2 <= MAX_M:
            candidates.add(m2)

    # Near LCM boundaries
    for k in range(0, max_val // LCM_12 + 1):
        m = k * LCM_12 - 1
        if 0 <= m <= MAX_M:
            candidates.add(m)
        m2 = k * LCM_12
        if 0 <= m2 <= MAX_M:
            candidates.add(m2)

    # Also try small m values
    for m in range(0, 50):
        candidates.add(m)

    # Sort candidates for deterministic processing
    candidates = sorted(candidates)

    best_m = 12
    best_f = 0

    for m in candidates:
        if time.time() > deadline:
            break
        f = _f_value(m, multiples, max_L=200)
        if f is not None and f > best_f:
            best_f = f
            best_m = m

    # If we still have time, do a finer local search around the best
    if time.time() < deadline - 5:
        for dm in range(-50, 51):
            m2 = best_m + dm
            if 0 <= m2 <= MAX_M and m2 not in candidates:
                if time.time() > deadline:
                    break
                f = _f_value(m2, multiples, max_L=200)
                if f is not None and f > best_f:
                    best_f = f
                    best_m = m2

    return int(best_m)
# EVOLVE-BLOCK-END
