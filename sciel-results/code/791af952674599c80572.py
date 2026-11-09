"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

P = {'n': 14, 'edges': [[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 10], [0, 12], [1, 4], [1, 9], [1, 11], [2, 3], [2, 5], [2, 6], [2, 7], [2, 12], [2, 13], [3, 4], [3, 5], [3, 8], [3, 11], [3, 12], [3, 13], [4, 5], [4, 7], [4, 13], [5, 8], [5, 9], [5, 10], [5, 12], [5, 13], [6, 8], [6, 11], [6, 13], [7, 12], [8, 9], [8, 13], [9, 11], [9, 12], [11, 13]]}

def construct():
    n = P['n']
    deadline = time.time() + 80.0  # safety margin before 90s kill
    
    # Build adjacency as bitmasks
    adj = [0] * n
    for u, v in P['edges']:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
    
    full_mask = (1 << n) - 1
    
    # Precompute which subsets are valid (clique or independent set)
    # A subset is a clique if every pair of vertices in it are adjacent
    # A subset is an independent set if no pair of vertices in it are adjacent
    valid = [False] * (1 << n)
    valid[0] = True  # empty set is trivially valid
    
    for mask in range(1, 1 << n):
        if time.time() > deadline:
            break
        # Find a vertex in the set
        lsb = mask & (-mask)
        v = lsb.bit_length() - 1
        rest = mask ^ lsb
        
        # Check if mask is a clique: v must be adjacent to all in rest
        is_clique = (adj[v] & rest) == rest
        # Check if mask is an independent set: v must not be adjacent to any in rest
        is_independent = (adj[v] & rest) == 0
        
        valid[mask] = is_clique or is_independent
    
    # DP: dp[mask] = minimum number of valid parts to partition 'mask'
    # We want dp[full_mask]
    INF = n + 1
    dp = [INF] * (1 << n)
    dp[0] = 0
    parent = [0] * (1 << n)  # parent[mask] = which valid subset was used to reach mask from parent
    
    for mask in range(1, 1 << n):
        if time.time() > deadline:
            break
        # Try all valid subsets that are subsets of mask and contain the lowest set bit
        # Optimization: only consider subsets containing the lowest set bit of mask
        lsb = mask & (-mask)
        # Enumerate all subsets of mask that contain lsb
        sub = mask
        while sub:
            if (sub & lsb) and valid[sub]:
                prev = mask ^ sub
                if dp[prev] + 1 < dp[mask]:
                    dp[mask] = dp[prev] + 1
                    parent[mask] = sub
            # Next subset containing lsb
            sub = (sub - 1) & mask
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break
    
    # Reconstruct the partition
    parts = []
    mask = full_mask
    while mask:
        sub = parent[mask]
        if sub == 0:
            # Should not happen if dp[full_mask] < INF, but safety
            sub = mask
        # Convert bitmask to list of vertices
        part = []
        m = sub
        while m:
            lsb2 = m & (-m)
            v = lsb2.bit_length() - 1
            part.append(v)
            m ^= lsb2
        parts.append(tuple(part))
        mask ^= sub
    
    # Convert to list of lists for return
    result = [list(p) for p in parts]
    return result

# EVOLVE-BLOCK-END
