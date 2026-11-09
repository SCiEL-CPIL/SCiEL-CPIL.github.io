import sys
import time

# EVOLVE-BLOCK-START
P = {'n': 14, 'edges': [[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 10], [0, 12], [1, 4], [1, 9], [1, 11], [2, 3], [2, 5], [2, 6], [2, 7], [2, 12], [2, 13], [3, 4], [3, 5], [3, 8], [3, 11], [3, 12], [3, 13], [4, 5], [4, 7], [4, 13], [5, 8], [5, 9], [5, 10], [5, 12], [5, 13], [6, 8], [6, 11], [6, 13], [7, 12], [8, 9], [8, 13], [9, 11], [9, 12], [11, 13]]}

def construct():
    n = P['n']
    edges = P['edges']
    
    # Build adjacency bitmasks
    adj = [0] * n
    for u, v in edges:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        
    # Precompute valid subsets
    # valid[mask] is True if mask represents a clique or an independent set
    valid = [False] * (1 << n)
    valid[0] = True
    
    for mask in range(1, 1 << n):
        # Check if clique
        is_clique = True
        # Check if independent
        is_independent = True
        
        # Extract vertices
        m = mask
        while m:
            lsb = m & -m
            v = lsb.bit_length() - 1
            m ^= lsb
            
            # For clique: all other vertices in mask must be neighbors of v
            others = mask ^ lsb
            if (adj[v] & others) != others:
                is_clique = False
                # Early break if not clique? We still need to check independent.
                # But if is_clique becomes False, we can't break the loop entirely 
                # unless we also know it's not independent, but checking independent 
                # is cheaper if we just check if adj[v] & others is 0.
                
            # For independent: no other vertex in mask should be a neighbor of v
            if adj[v] & others:
                is_independent = False
                
            # Optimization: if both are False, break
            if not is_clique and not is_independent:
                break
                
        if is_clique or is_independent:
            valid[mask] = True

    # DP to find minimum number of parts
    INF = 10**9
    dp = [INF] * (1 << n)
    dp[0] = 0
    # parent[mask] stores the last subset added to form mask
    parent = [-1] * (1 << n)
    
    for mask in range(1, 1 << n):
        # Iterate over all submasks of mask
        sub = mask
        while sub:
            if valid[sub]:
                prev = mask ^ sub
                if dp[prev] + 1 < dp[mask]:
                    dp[mask] = dp[prev] + 1
                    parent[mask] = sub
            sub = (sub - 1) & mask
            
    full_mask = (1 << n) - 1
    if dp[full_mask] == INF:
        # Fallback to greedy if something went wrong (should not happen)
        return [[i] for i in range(n)]
        
    # Reconstruct partition
    parts = []
    mask = full_mask
    while mask:
        sub = parent[mask]
        # Convert sub bitmask to list of vertices
        part_vertices = []
        s = sub
        while s:
            lsb = s & -s
            v = lsb.bit_length() - 1
            part_vertices.append(v)
            s ^= lsb
        parts.append(part_vertices)
        mask ^= sub
        
    return parts

# EVOLVE-BLOCK-END
