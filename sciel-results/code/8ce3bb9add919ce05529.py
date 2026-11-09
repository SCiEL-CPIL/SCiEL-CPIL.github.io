"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def _precompute_divisibility():
    """For each residue r in 0..27719 and offset j in 0..143, 
    compute bitmask of which i (1..12) divide (r+1+j)."""
    P = 27720
    MAX_L = 144
    div_mask = [[0] * MAX_L for _ in range(P)]
    for r in range(P):
        for j in range(MAX_L):
            val = r + 1 + j
            mask = 0
            for i in range(1, 13):
                if val % i == 0:
                    mask |= (1 << (i - 1))
            div_mask[r][j] = mask
    return div_mask

def _has_matching(div_mask_r, L):
    """Check if there's a matching of size 12 in window of length L.
    div_mask_r[j] has bit i-1 set if i divides position j (0-indexed).
    We need to assign each i (1..12) to a distinct position j in 0..L-1 
    such that i divides position j's value."""
    # Build adjacency: adj[i] is a list of positions j in [0, L) where i divides (r+1+j)
    adj = [[] for _ in range(13)]  # adj[i] = list of positions
    for j in range(L):
        mask = div_mask_r[j]
        for i in range(1, 13):
            if mask & (1 << (i - 1)):
                adj[i].append(j)
    
    # Hopcroft-Karp bipartite matching
    # Left nodes: i in 1..12, Right nodes: positions 0..L-1
    pair_u = [-1] * 13  # pair_u[i] = position matched to i, or -1
    pair_v = [-1] * L   # pair_v[j] = i matched to position j, or -1
    dist = [0] * 13
    
    from collections import deque
    
    def bfs():
        """Find augmenting paths from all free left nodes"""
        queue = deque()
        found = False
        for i in range(1, 13):
            if pair_u[i] == -1:
                dist[i] = 0
                queue.append(i)
            else:
                dist[i] = -1
        while queue:
            i = queue.popleft()
            for j in adj[i]:
                matched_i = pair_v[j]
                if matched_i == -1:
                    found = True
                elif dist[matched_i] == -1:
                    dist[matched_i] = dist[i] + 1
                    queue.append(matched_i)
        return found
    
    def dfs(i):
        """Try to find augmenting path from left node i"""
        for j in adj[i]:
            matched_i = pair_v[j]
            if matched_i == -1 or (dist[matched_i] == dist[i] + 1 and dfs(matched_i)):
                pair_u[i] = j
                pair_v[j] = i
                return True
        dist[i] = -1
        return False
    
    matching_size = 0
    while bfs():
        for i in range(1, 13):
            if pair_u[i] == -1 and dfs(i):
                matching_size += 1
    
    return matching_size == 12

def _compute_f(r, div_mask):
    """Compute f(12, m) where m mod 27720 = r using binary search on L."""
    # f(12, m) is the least L such that matching exists in window of length L
    # This is monotone: if matching exists for L, it exists for L+1
    lo, hi = 1, 144
    while lo < hi:
        mid = (lo + hi) // 2
        if _has_matching(div_mask[r], mid):
            hi = mid
        else:
            lo = mid + 1
    return lo

def construct():
    start_time = time.time()
    DEADLINE = 85  # safety margin before 90s kill
    
    div_mask = _precompute_divisibility()
    
    best_m = 12  # fallback
    best_f = 12
    
    for r in range(27720):
        if time.time() - start_time > DEADLINE:
            break
        f_val = _compute_f(r, div_mask)
        if f_val > best_f:
            best_f = f_val
            # m = r works (r is in 0..27719, which is in 0..1000000)
            best_m = r
    
    return int(best_m)
# EVOLVE-BLOCK-END
