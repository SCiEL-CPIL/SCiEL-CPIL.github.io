"""Rewritten program with exact bitmask DP for cochromatic partition."""
import time

P = {'n': 14, 'edges': [[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 10], [0, 12], [1, 4], [1, 9], [1, 11], [2, 3], [2, 5], [2, 6], [2, 7], [2, 12], [2, 13], [3, 4], [3, 5], [3, 8], [3, 11], [3, 12], [3, 13], [4, 5], [4, 7], [4, 13], [5, 8], [5, 9], [5, 10], [5, 12], [5, 13], [6, 8], [6, 11], [6, 13], [7, 12], [8, 9], [8, 13], [9, 11], [9, 12], [11, 13]]}

def construct():
    deadline = time.time() + 80.0
    
    n = P['n']
    edges = P['edges']
    
    # Build adjacency masks
    adj_mask = [0] * n
    for u, v in edges:
        adj_mask[u] |= (1 << v)
        adj_mask[v] |= (1 << u)
        
    full_mask = (1 << n) - 1
    size = 1 << n
    
    # Precompute is_clique and is_independent for all masks
    is_clique = [False] * size
    is_independent = [False] * size
    is_clique[0] = True
    is_independent[0] = True
    
    for mask in range(1, size):
        # Get lowest set bit
        lbit = mask & -mask
        v = lbit.bit_length() - 1
        rest = mask ^ lbit
        
        # Check clique
        # rest must be clique, and v must be connected to all in rest
        if is_clique[rest] and (adj_mask[v] & rest) == rest:
            is_clique[mask] = True
            
        # Check independent set
        # rest must be independent, and v must NOT be connected to any in rest
        if is_independent[rest] and (adj_mask[v] & rest) == 0:
            is_independent[mask] = True
            
    # valid[mask] is True if mask is non-empty and (clique or independent)
    valid = [False] * size
    for mask in range(1, size):
        if is_clique[mask] or is_independent[mask]:
            valid[mask] = True
            
    # DP
    INF = 10**9
    dp = [INF] * size
    parent = [-1] * size
    dp[0] = 0
    
    for mask in range(1, size):
        if time.time() > deadline:
            break
        lbit = mask & -mask
        sub = mask
        while sub:
            if (sub & lbit) and valid[sub]:
                prev = mask ^ sub
                if dp[prev] + 1 < dp[mask]:
                    dp[mask] = dp[prev] + 1
                    parent[mask] = sub
            sub = (sub - 1) & mask
            
    # If DP failed (timeout), fallback to greedy
    if dp[full_mask] == INF:
        parts = []
        for v in range(n):
            parts.append([v])
        return parts

    # Backtrack to find the partition
    current = full_mask
    parts_masks = []
    while current > 0:
        p = parent[current]
        if p == -1:
             break
        parts_masks.append(p)
        current ^= p
        
    # Convert masks to lists of vertices
    result = []
    for mask in parts_masks:
        verts = []
        for v in range(n):
            if mask & (1 << v):
                verts.append(v)
        result.append(verts)
        
    return result
