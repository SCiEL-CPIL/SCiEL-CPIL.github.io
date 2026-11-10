"""Find m in [0,1000000] maximizing f(12,m), the least L with a distinct-multiple matching in (m,m+L]."""
import time
from math import gcd

# EVOLVE-BLOCK-START
N = 12
MAX_START = 1000000
PERIOD = 27720  # lcm(1..12)

# Divisors of 27720 that are <= 12: 1,2,3,4,6,7,11,12
_DIVS = [1, 2, 3, 4, 6, 7, 11, 12]

def _hk_match(adj):
    """Hopcroft-Karp on left nodes 0..11 -> right nodes 0..11 via adj. Returns matching size."""
    n = 12
    pair_u = [-1] * n
    pair_v = [-1] * n
    dist = [0] * n

    def bfs():
        from collections import deque
        q = deque()
        for u in range(n):
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
        for u in range(n):
            if pair_u[u] == -1 and dfs(u):
                matching += 1
    return matching

def _f_for_residue(r):
    """Compute f(12, r) = least L>0 with a perfect matching in (r, r+L]."""
    def feasible(L):
        # adj[i] = list of divisors d that have a multiple in (r, r+L]
        adj = [[] for _ in range(12)]
        for i in range(1, 13):
            if ((r + L) // i) > (r // i):
                adj[i - 1] = _DIVS
        return _hk_match(adj) == 12
    # f is at most 144; binary search the least feasible L
    lo, hi = 1, 144
    # ensure hi feasible
    while not feasible(hi):
        hi *= 2
    while lo < hi:
        mid = (lo + hi) // 2
        if feasible(mid):
            hi = mid
        else:
            lo = mid + 1
    return lo

def construct():
    t0 = time.time()
    deadline = 80.0  # safety margin under 90s wall-clock

    # Find the residue class with the maximum f
    best_f = 0
    best_r = 0
    for r in range(PERIOD):
        if time.time() > deadline:
            break
        f = _f_for_residue(r)
        if f > best_f:
            best_f = f
            best_r = r
            if best_f >= 144:  # global upper bound reached
                break

    # Smallest m in [0, MAX_START] with m % PERIOD == best_r
    m = best_r
    if m > MAX_START:
        m = 0
    return int(m)
# EVOLVE-BLOCK-END
