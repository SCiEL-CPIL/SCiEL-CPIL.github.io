import time
import sys

P = {'n': 14, 'edges': [[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 10], [0, 12], [1, 4], [1, 9], [1, 11], [2, 3], [2, 5], [2, 6], [2, 7], [2, 12], [2, 13], [3, 4], [3, 5], [3, 8], [3, 11], [3, 12], [3, 13], [4, 5], [4, 7], [4, 13], [5, 8], [5, 9], [5, 10], [5, 12], [5, 13], [6, 8], [6, 11], [6, 13], [7, 12], [8, 9], [8, 13], [9, 11], [9, 12], [11, 13]]}

def construct():
    n = P['n']
    edges = P['edges']
    
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin before 90s kill
    
    # Build adjacency masks
    adj = [0] * n
    for u, v in edges:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        
    # Precompute validity of all subsets
    num_subsets = 1 << n
    valid_part = [False] * num_subsets
    valid_part[0] = True
    
    for mask in range(1, num_subsets):
        is_cli = True
        is_ind = True
        m = mask
        while m:
            lsb = m & -m
            v = lsb.bit_length() - 1
            if (adj[v] & mask) != (mask ^ lsb):
                is_cli = False
            if (adj[v] & mask) != 0:
                is_ind = False
            if not is_cli and not is_ind:
                break
            m ^= lsb
        if is_cli or is_ind:
            valid_part[mask] = True
    
    # Collect all valid non-empty subsets
    valid_masks = [m for m in range(1, num_subsets) if valid_part[m]]
    
    # DP: min parts to cover subset 'mask'
    # dp[mask] = minimum number of valid parts that partition 'mask'
    INF = n + 1
    dp = [INF] * num_subsets
    dp[0] = 0
    parent = [-1] * num_subsets  # parent[mask] = the valid part mask used to reach mask
    
    all_vertices_mask = (1 << n) - 1
    
    # Iterate over all masks, for each try adding a valid part
    # For efficiency, we iterate masks in increasing order of popcount
    # Actually, we can just iterate all masks and for each, try all valid parts that are subsets
    # But that's O(3^n) which is too slow.
    
    # Better: for each mask, dp[mask] = min over valid part p subset of mask of (dp[mask ^ p] + 1)
    # We can iterate masks in order and for each mask, iterate valid parts p that are subsets of mask.
    # To do this efficiently, we can iterate over valid parts p and update supersets.
    # Or: iterate mask from 1 to num_subsets-1, and for each valid part p subset of mask, update dp[mask].
    # Number of valid parts is at most 2^n, and for each mask, number of valid part subsets could be large.
    
    # Alternative: iterate valid parts, and for each valid part p, update dp for all masks that contain p.
    # dp[mask | p] = min(dp[mask | p], dp[mask] + 1) where mask & p == 0
    # This is like a knapsack-style DP.
    
    # Let's do: for each valid part p, for each mask from 0 to num_subsets-1:
    #   if mask & p == 0 and dp[mask] < INF:
    #       new_mask = mask | p
    #       if dp[mask] + 1 < dp[new_mask]:
    #           dp[new_mask] = dp[mask] + 1
    #           parent[new_mask] = p
    
    # But this is O(num_valid_parts * num_subsets) which could be up to 2^n * 2^n = 2^(2n) = 2^28 ~ 268M, too slow.
    
    # Better approach: iterate masks in increasing order. For each mask, find all valid parts p that are subsets of mask.
    # dp[mask] = min(dp[mask ^ p] + 1) for all valid p subset of mask.
    # To enumerate valid subsets p of mask, we can iterate submasks of mask and check if valid.
    # Number of submasks of mask is 2^(popcount(mask)). Total work is sum over masks of 2^(popcount(mask)) = 3^n.
    # 3^14 = 4,782,969, which is feasible!
    
    for mask in range(1, num_subsets):
        # Enumerate all submasks of mask
        sub = mask
        while sub:
            if valid_part[sub]:
                prev = mask ^ sub
                if dp[prev] + 1 < dp[mask]:
                    dp[mask] = dp[prev] + 1
                    parent[mask] = sub
            sub = (sub - 1) & mask
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break
    
    # Reconstruct solution
    if dp[all_vertices_mask] >= INF:
        # Fallback: each vertex its own part
        result = [[i] for i in range(n)]
        return result
    
    parts_masks = []
    cur = all_vertices_mask
    while cur:
        p = parent[cur]
        if p == -1:
            break
        parts_masks.append(p)
        cur ^= p
    
    # Convert to list of lists
    result = []
    for pm in parts_masks:
        part_vertices = []
        for v in range(n):
            if pm & (1 << v):
                part_vertices.append(v)
        result.append(part_vertices)
    
    return result
