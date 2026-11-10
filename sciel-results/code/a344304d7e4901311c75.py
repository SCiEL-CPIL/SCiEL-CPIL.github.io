"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

P = {'n': 14, 'edges': [[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 10], [0, 12], [1, 4], [1, 9], [1, 11], [2, 3], [2, 5], [2, 6], [2, 7], [2, 12], [2, 13], [3, 4], [3, 5], [3, 8], [3, 11], [3, 12], [3, 13], [4, 5], [4, 7], [4, 13], [5, 8], [5, 9], [5, 10], [5, 12], [5, 13], [6, 8], [6, 11], [6, 13], [7, 12], [8, 9], [8, 13], [9, 11], [9, 12], [11, 13]]}

def construct():
    n = P['n']
    adj = [0] * n
    for a, b in P['edges']:
        adj[a] |= (1 << b)
        adj[b] |= (1 << a)
    
    # Precompute clique and independent set properties for all masks
    # is_clique[mask] and is_independent[mask]
    size = 1 << n
    is_clique = [False] * size
    is_independent = [False] * size
    is_clique[0] = True
    is_independent[0] = True
    
    for mask in range(1, size):
        # Find least significant set bit
        lsb = mask & -mask
        v = (lsb.bit_length() - 1)
        rest = mask ^ lsb
        
        # Check clique: v must be connected to all in rest, and rest must be clique
        is_clique[mask] = is_clique[rest] and ((adj[v] & rest) == rest)
        
        # Check independent set: v must not be connected to any in rest, and rest must be independent
        is_independent[mask] = is_independent[rest] and ((adj[v] & rest) == 0)
        
        # Early exit optimization not strictly necessary for n=14, but good practice
        # if not is_clique[mask] and not is_independent[mask]: pass # No change needed, just store
    
    is_valid = [is_clique[m] or is_independent[m] for m in range(size)]
    
    # Exact O(3^n) dynamic programming for cochromatic number
    # dp[mask] = minimum number of parts to partition vertices in mask
    # where each part must be a clique or independent set
    
    full_mask = (1 << n) - 1
    INF = n + 1  # Upper bound on parts
    
    dp = [INF] * (1 << n)
    dp[0] = 0
    parent = [0] * (1 << n)  # Store which sub-part was used to reach this state
    
    # Precompute list of valid subsets (cliques or independent sets)
    # For each mask, we need to find its valid submasks efficiently
    
    for mask in range(1, 1 << n):
        # Find the least significant bit
        lsb = mask & -mask
        v = lsb.bit_length() - 1
        
        # Try all valid submasks of mask that contain vertex v
        # This ensures we don't double-count partitions
        sub = mask
        while sub:
            if sub & lsb:  # Submask must contain v to avoid duplicates
                if is_valid[sub]:
                    rest = mask ^ sub
                    if dp[rest] + 1 < dp[mask]:
                        dp[mask] = dp[rest] + 1
                        parent[mask] = sub
            sub = (sub - 1) & mask
    
    # Reconstruct the optimal partition
    result = []
    mask = full_mask
    while mask:
        sub = parent[mask]
        part_vertices = []
        m = sub
        while m:
            lsb = m & -m
            v = lsb.bit_length() - 1
            part_vertices.append(v)
            m ^= lsb
        result.append(part_vertices)
        mask ^= sub
    
    return result
# EVOLVE-BLOCK-END
