import time
import sys

P = {'n': 14, 'edges': [[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 10], [0, 12], [1, 4], [1, 9], [1, 11], [2, 3], [2, 5], [2, 6], [2, 7], [2, 12], [2, 13], [3, 4], [3, 5], [3, 8], [3, 11], [3, 12], [3, 13], [4, 5], [4, 7], [4, 13], [5, 8], [5, 9], [5, 10], [5, 12], [5, 13], [6, 8], [6, 11], [6, 13], [7, 12], [8, 9], [8, 13], [9, 11], [9, 12], [11, 13]]}

def construct():
    n = P['n']
    edges = P['edges']
    
    # Build adjacency as bitmasks
    adj = [0] * n
    for u, v in edges:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
    
    all_mask = (1 << n) - 1
    
    # Precompute which subsets are cliques and which are independent sets
    # A subset is a clique if every pair of vertices in it is adjacent
    # A subset is an independent set if no pair of vertices in it is adjacent
    
    is_clique = [False] * (1 << n)
    is_independent = [False] * (1 << n)
    
    # Single vertex subsets are both cliques and independent sets
    for v in range(n):
        mask = 1 << v
        is_clique[mask] = True
        is_independent[mask] = True
    
    # Empty set is both (but we won't use empty sets in partition)
    is_clique[0] = True
    is_independent[0] = True
    
    # For subsets with 2+ vertices, check if clique or independent
    # Use DP: subset S is a clique if S\{v} is a clique and v is adjacent to all in S\{v}
    # subset S is independent if S\{v} is independent and v is non-adjacent to all in S\{v}
    
    for mask in range(1, 1 << n):
        if mask & (mask - 1) == 0:
            continue  # single vertex, already handled
        
        # Get lowest set bit
        lb = mask & (-mask)
        v = lb.bit_length() - 1
        rest = mask ^ lb
        
        # Check clique: rest is clique AND v adjacent to all in rest
        if is_clique[rest]:
            # Check if v is adjacent to all vertices in rest
            if adj[v] & rest == rest:
                is_clique[mask] = True
        
        # Check independent: rest is independent AND v non-adjacent to all in rest
        if is_independent[rest]:
            # Check if v is NOT adjacent to any vertex in rest
            if adj[v] & rest == 0:
                is_independent[mask] = True
    
    # Now do subset DP to find minimum number of parts
    # dp[mask] = minimum number of parts to cover exactly the vertices in mask
    INF = float('inf')
    dp = [INF] * (1 << n)
    dp[0] = 0
    
    # For each mask, try all submasks that are valid parts (clique or independent)
    for mask in range(1, 1 << n):
        # Enumerate all submasks of mask
        sub = mask
        while sub > 0:
            if is_clique[sub] or is_independent[sub]:
                if dp[mask ^ sub] + 1 < dp[mask]:
                    dp[mask] = dp[mask ^ sub] + 1
            sub = (sub - 1) & mask
    
    # dp[all_mask] gives the minimum number of parts
    min_parts = dp[all_mask]
    
    # Now reconstruct the actual partition
    # We need to track which submask was chosen at each step
    # Redo DP with backtracking info
    parent = [0] * (1 << n)  # parent[mask] = the submask chosen for mask
    dp2 = [INF] * (1 << n)
    dp2[0] = 0
    parent[0] = 0
    
    for mask in range(1, 1 << n):
        sub = mask
        while sub > 0:
            if is_clique[sub] or is_independent[sub]:
                if dp2[mask ^ sub] + 1 < dp2[mask]:
                    dp2[mask] = dp2[mask ^ sub] + 1
                    parent[mask] = sub
            sub = (sub - 1) & mask
    
    # Reconstruct partition
    parts = []
    mask = all_mask
    while mask > 0:
        sub = parent[mask]
        # Extract vertices from sub
        part_verts = []
        for v in range(n):
            if sub & (1 << v):
                part_verts.append(v)
        parts.append(part_verts)
        mask ^= sub
    
    # Return as list of lists of ints
    return parts
