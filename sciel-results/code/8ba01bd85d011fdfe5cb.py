"""Cochromatic partition via optimized subset DP with submask enumeration."""
import time

P = {'n': 14, 'edges': [[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 10], [0, 12], [1, 4], [1, 9], [1, 11], [2, 3], [2, 5], [2, 6], [2, 7], [2, 12], [2, 13], [3, 4], [3, 5], [3, 8], [3, 11], [3, 12], [3, 13], [4, 5], [4, 7], [4, 13], [5, 8], [5, 9], [5, 10], [5, 12], [5, 13], [6, 8], [6, 11], [6, 13], [7, 12], [8, 9], [8, 13], [9, 11], [9, 12], [11, 13]]}

def construct():
    start_time = time.time()
    deadline = 75.0  # safety margin before 90s kill
    
    n = P['n']
    full_mask = (1 << n) - 1
    size = 1 << n
    
    # Build adjacency bitmasks
    adj = [0] * n
    for u, v in P['edges']:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
    
    # Precompute is_indep and is_clique for all masks using DP
    is_indep = [False] * size
    is_clique = [False] * size
    is_indep[0] = True
    is_clique[0] = True
    
    for mask in range(1, size):
        lb = mask & (-mask)
        v = lb.bit_length() - 1
        rest = mask ^ lb
        is_indep[mask] = is_indep[rest] and (adj[v] & rest) == 0
        is_clique[mask] = is_clique[rest] and (adj[v] & rest) == rest
    
    # is_valid: mask is a non-empty clique or independent set
    is_valid = [False] * size
    for mask in range(1, size):
        is_valid[mask] = is_indep[mask] or is_clique[mask]
    
    # DP: dp[mask] = min number of parts to partition vertices in mask
    INF = n + 1
    dp = [INF] * size
    parent = [0] * size
    dp[0] = 0
    
    for mask in range(1, size):
        # Check deadline
        if time.time() - start_time > deadline:
            # Fallback: reconstruct best effort with what we have
            parts = []
            current = full_mask
            while current:
                if parent[current]:
                    part_mask = parent[current]
                    verts = [i for i in range(n) if (part_mask >> i) & 1]
                    parts.append(verts)
                    current ^= part_mask
                else:
                    lb = current & (-current)
                    v = lb.bit_length() - 1
                    parts.append([v])
                    current ^= lb
            return [[int(v) for v in part] for part in parts]
        
        if is_valid[mask]:
            dp[mask] = 1
            parent[mask] = mask
            continue
        
        best_val = INF
        best_sub = 0
        
        # Optimization: Only consider submasks containing the lowest set bit of mask.
        # This avoids redundant permutations of the partition parts.
        lb = mask & (-mask)
        rest_mask = mask ^ lb
        sub_rest = rest_mask
        
        while True:
            sub = sub_rest | lb
            if is_valid[sub]:
                rest = mask ^ sub
                val = dp[rest] + 1
                if val < best_val:
                    best_val = val
                    best_sub = sub
                    if best_val == 2:
                        break
            if sub_rest == 0:
                break
            sub_rest = (sub_rest - 1) & rest_mask
        
        dp[mask] = best_val
        parent[mask] = best_sub
    
    # Reconstruct the optimal partition
    parts = []
    current = full_mask
    while current:
        part_mask = parent[current]
        if part_mask == 0:
            # Should not happen, but fallback
            lb = current & (-current)
            v = lb.bit_length() - 1
            parts.append([v])
            current ^= lb
        else:
            verts = [i for i in range(n) if (part_mask >> i) & 1]
            parts.append(verts)
            current ^= part_mask
    
    # Ensure types: list of lists of Python ints
    return [[int(v) for v in part] for part in parts]
