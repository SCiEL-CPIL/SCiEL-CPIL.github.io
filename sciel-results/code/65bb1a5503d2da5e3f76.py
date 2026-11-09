"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd

def _lcm(a, b):
    return a // gcd(a, b) * b

def _compute_lcm_range(n):
    result = 1
    for i in range(2, n + 1):
        result = _lcm(result, i)
    return result

def _has_perfect_matching(n, adj, pool_max):
    # adj[i] = list of candidate indices (0-based) that i can match to
    # pool_max: only consider candidates with index < pool_max
    # We use DFS-based bipartite matching (Hopcroft-Karp would be overkill for n=12)
    match_to = [-1] * pool_max  # which i is matched to candidate j
    match_from = [-1] * n       # which candidate j is matched to i

    def dfs(i, seen):
        for j in adj[i]:
            if j >= pool_max:
                continue
            if seen[j]:
                continue
            seen[j] = True
            if match_to[j] == -1 or dfs(match_to[j], seen):
                match_to[j] = i
                match_from[i] = j
                return True
        return False

    matched = 0
    for i in range(n):
        seen = [False] * pool_max
        if dfs(i, seen):
            matched += 1
    return matched == n

def _compute_f(r, n, max_L, adj):
    # adj[i] is sorted list of candidate indices (0-based, candidate value = r+1+idx)
    # For a given L, pool_max = L (candidates with index 0..L-1 are in (r, r+L])
    # Actually candidate idx corresponds to value r+1+idx, which is in (r, r+L] iff idx < L
    for L in range(1, max_L + 1):
        if _has_perfect_matching(n, adj, L):
            return L
    return max_L

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # safety margin before 90s kill

    n = 12
    max_L = 144
    period = _compute_lcm_range(n)  # 27720

    # Precompute adjacency for each residue r in [0, period)
    # adj[r][i] = sorted list of candidate indices j such that (r+1+j) % i == 0
    # Candidate values are r+1, r+2, ..., r+max_L (indices 0..max_L-1)

    best_r = 0
    best_f = 0

    for r in range(period):
        if time.time() > deadline:
            break
        # Build adjacency for this residue
        adj = [[] for _ in range(n)]
        for i in range(1, n + 1):
            # candidates r+1+j for j in 0..max_L-1, need (r+1+j) % i == 0
            # j such that (r+1+j) ≡ 0 (mod i) => j ≡ -(r+1) (mod i)
            # j in [0, max_L-1]
            # Find first j >= 0 with (r+1+j) % i == 0
            rem = (r + 1) % i
            if rem == 0:
                first_j = 0
            else:
                first_j = i - rem
            for j in range(first_j, max_L, i):
                adj[i - 1].append(j)

        f_val = _compute_f(r, n, max_L, adj)
        if f_val > best_f:
            best_f = f_val
            best_r = r

    # Now find m in [0, 1000000] with m % period == best_r
    # m = best_r + k * period <= 1000000
    max_k = (1000000 - best_r) // period
    if max_k < 0:
        m = best_r  # should not happen since best_r < 27720 <= 1000000
    else:
        m = best_r + max_k * period
    # Ensure m is a Python int
    m = int(m)
    return m
# EVOLVE-BLOCK-END
