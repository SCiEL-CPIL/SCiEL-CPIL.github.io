"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

P = {'n': 14, 'edges': [[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 10], [0, 12], [1, 4], [1, 9], [1, 11], [2, 3], [2, 5], [2, 6], [2, 7], [2, 12], [2, 13], [3, 4], [3, 5], [3, 8], [3, 11], [3, 12], [3, 13], [4, 5], [4, 7], [4, 13], [5, 8], [5, 9], [5, 10], [5, 12], [5, 13], [6, 8], [6, 11], [6, 13], [7, 12], [8, 9], [8, 13], [9, 11], [9, 12], [11, 13]]}

def construct():
    t0 = time.time()
    DEADLINE = 85.0  # safety margin before 90s kill
    n = P['n']
    edges = P['edges']
    
    # Build adjacency bitmasks
    adj = [0] * n
    for u, v in edges:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
    
    # Precompute valid_mask: True if subset is clique or independent set
    # Use bit operations for speed
    size = 1 << n
    valid_mask = [False] * size
    valid_mask[0] = True
    
    # For each mask, check clique/independent using bit tricks
    # A set S is a clique iff for every v in S, S \ {v} subset of adj[v]
    # A set S is independent iff for every v in S, S \ {v} subset of ~adj[v]
    
    # Precompute popcount and use incremental approach
    # For each mask, pick lowest bit, check if removing it keeps clique/ind property
    # is_clique[mask]: mask is a clique
    # is_ind[mask]: mask is an independent set
    
    is_clique = [False] * size
    is_ind = [False] * size
    is_clique[0] = True
    is_ind[0] = True
    
    for mask in range(1, size):
        # Get lowest set bit
        lb = mask & (-mask)
        v = lb.bit_length() - 1
        rest = mask ^ lb
        # mask is clique iff rest is clique AND rest subset of adj[v]
        is_clique[mask] = is_clique[rest] and (rest & ~adj[v]) == 0
        # mask is independent iff rest is independent AND rest subset of ~adj[v]
        # ~adj[v] within n bits
        complement_adj = (size - 1) ^ adj[v]
        is_ind[mask] = is_ind[rest] and (rest & ~complement_adj) == 0
        valid_mask[mask] = is_clique[mask] or is_ind[mask]
    
    # Precompute list of valid submasks for each mask to avoid re-checking
    # Actually, let's just use the valid_mask array directly in the DP
    
    # DP: dp[mask] = min number of parts to cover vertices in 'mask'
    # To speed up, iterate masks in order and for each, try valid submasks
    # Key optimization: only iterate submasks that are valid
    
    # Alternative faster approach: for each mask, find the minimum by
    # iterating over valid submasks. We can precompute for each mask the list
    # of valid submasks, but that's memory heavy. Instead, we iterate submasks
    # and check valid_mask[sub].
    
    # Better: iterate in a smarter order. For each mask, we want min over valid submasks s of mask:
    # dp[mask ^ s] + 1. 
    # We can iterate s over submasks of mask, but skip invalid ones quickly.
    
    # Even better: note that if we fix the "first" vertex in the partition,
    # we can reduce the search. But let's stick with the DP and optimize.
    
    # Optimization: use the fact that we only need to consider submasks that
    # are valid AND contain the lowest set bit of mask (WLOG, the first part
    # in our canonical ordering contains the lowest vertex). This reduces
    # the submask enumeration significantly.
    
    dp = [255] * size  # use 255 as infinity (max parts is n=14)
    dp[0] = 0
    choice = [0] * size
    
    for mask in range(1, size):
        if time.time() - t0 > DEADLINE:
            break
        # Find lowest set bit
        lb = mask & (-mask)
        best = 255
        best_sub = 0
        # Iterate submasks of mask that contain lb
        # submasks of (mask ^ lb) are the "rest", then sub = rest | lb
        rest_mask = mask ^ lb
        sub = rest_mask
        while True:
            candidate = sub | lb
            if valid_mask[candidate]:
                prev = mask ^ candidate
                val = dp[prev] + 1
                if val < best:
                    best = val
                    best_sub = candidate
            if sub == 0:
                break
            sub = (sub - 1) & rest_mask
        dp[mask] = best
        choice[mask] = best_sub
    
    full_mask = size - 1
    min_parts = dp[full_mask]
    
    # Reconstruct
    parts = []
    mask = full_mask
    while mask:
        sub = choice[mask]
        if sub == 0:
            # Fallback: single vertex
            sub = mask & (-mask)
        vertices = [i for i in range(n) if sub & (1 << i)]
        parts.append(vertices)
        mask ^= sub
    
    result = [list(part) for part in parts]
    return result
# EVOLVE-BLOCK-END
