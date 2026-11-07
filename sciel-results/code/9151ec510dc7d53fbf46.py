import time
import math

# EVOLVE-BLOCK-START
# Strategy:
# The condition for f(12, m) depends on the residues of m+1, ..., m+L modulo 1, ..., 12.
# Since LCM(1, ..., 12) = 27720, the pattern of divisibility repeats every 27720.
# Specifically, the set of valid matchings in (m, m+L] depends only on m mod 27720.
# Therefore, f(12, m) is periodic with period 27720.
# We can compute f(12, r) for all r in [0, 27720) and find the r that maximizes f(12, r).
# Then we choose the largest m in [0, 1000000] such that m % 27720 == r_best.
# This guarantees the maximum possible value of f(12, m) for any m in range.

PERIOD = 27720
N = 12
MAX_M = 1000000

def has_valid_set_fast(m, L):
    """
    Check if there is a perfect matching for indices 1..12 to distinct numbers
    in (m, m+L] such that i divides b_i.
    Uses Hopcroft-Karp algorithm for O(E*sqrt(V)) performance.
    """
    # Precompute neighbors for each i in 1..12
    # neighbors[i-1] is a list of j such that (m+1+j) % i == 0
    neighbors = [[] for _ in range(N)]
    start = m + 1
    for i in range(1, N + 1):
        rem = (-start) % i
        if rem < L:
            neighbors[i - 1].extend(range(rem, L, i))
    
    # Hopcroft-Karp implementation
    INF = float('inf')
    pair_u = [-1] * N      # pair_u[u] = right node matched to left node u, or -1
    pair_v = [-1] * L      # pair_v[v] = left node matched to right node v, or -1
    dist = [0] * N
    
    def bfs():
        from collections import deque
        queue = deque()
        for u in range(N):
            if pair_u[u] == -1:
                dist[u] = 0
                queue.append(u)
            else:
                dist[u] = INF
        found = False
        while queue:
            u = queue.popleft()
            for v in neighbors[u]:
                pu = pair_v[v]
                if pu == -1:
                    found = True
                elif dist[pu] == INF:
                    dist[pu] = dist[u] + 1
                    queue.append(pu)
        return found
    
    def dfs(u):
        for v in neighbors[u]:
            pu = pair_v[v]
            if pu == -1 or (dist[pu] == dist[u] + 1 and dfs(pu)):
                pair_u[u] = v
                pair_v[v] = u
                return True
        dist[u] = INF
        return False
    
    matching = 0
    while bfs():
        for u in range(N):
            if pair_u[u] == -1:
                if dfs(u):
                    matching += 1
    
    return matching == N

def f_12(m):
    """
    Compute the least positive L such that (m, m+L] has a valid set.
    """
    # Lower bound is 12 (need 12 distinct numbers)
    # Upper bound is 144
    lo, hi = 12, 144
    while lo < hi:
        mid = (lo + hi) // 2
        if has_valid_set_fast(m, mid):
            hi = mid
        else:
            lo = mid + 1
    return lo

def construct():
    start_time = time.time()
    
    # 1. Find the best residue r in [0, PERIOD)
    best_r = 0
    best_val = 0
    
    # Iterate through all residues
    for r in range(PERIOD):
        if time.time() - start_time > 80: # Safety margin (limit 90s)
            # If we time out, return the best found so far
            # We must ensure best_r is valid
            break
        
        val = f_12(r)
        if val > best_val:
            best_val = val
            best_r = r
    
    # 2. Map best_r to the largest m in [0, MAX_M] with m % PERIOD == best_r
    if best_r > MAX_M:
        m = best_r
    else:
        k_max = (MAX_M - best_r) // PERIOD
        m = k_max * PERIOD + best_r
        
    # Ensure m is within bounds
    if m < 0:
        m = 0
    if m > MAX_M:
        m = MAX_M
        
    # Verify the result is a valid integer
    if not isinstance(m, int):
        m = int(m)
        
    return m
# EVOLVE-BLOCK-END
