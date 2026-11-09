"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

P = {'n': 14, 'edges': [[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 10], [0, 12], [1, 4], [1, 9], [1, 11], [2, 3], [2, 5], [2, 6], [2, 7], [2, 12], [2, 13], [3, 4], [3, 5], [3, 8], [3, 11], [3, 12], [3, 13], [4, 5], [4, 7], [4, 13], [5, 8], [5, 9], [5, 10], [5, 12], [5, 13], [6, 8], [6, 11], [6, 13], [7, 12], [8, 9], [8, 13], [9, 11], [9, 12], [11, 13]]}

def construct():
    deadline = time.time() + 80  # safety margin before 90s kill
    n = P['n']
    full_mask = (1 << n) - 1
    
    # Build adjacency bitmask for each vertex
    adj = [0] * n
    edge_set = set()
    for u, v in P['edges']:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        edge_set.add((min(u,v), max(u,v)))
    
    # Precompute: for each subset (bitmask), is it a clique or independent set?
    # A subset is a clique if every pair has an edge
    # A subset is an independent set if no pair has an edge
    
    # Use DP: valid_clique[mask], valid_indep[mask]
    # A mask is a clique iff for every vertex v in mask, (mask without v) is a clique
    # and v is adjacent to all other vertices in mask
    # Similarly for independent set: v is NOT adjacent to any other vertex in mask
    
    valid = [False] * (1 << n)
    valid[0] = True  # empty set is both
    
    for mask in range(1, 1 << n):
        # Find lowest set bit
        lsb = mask & (-mask)
        v = lsb.bit_length() - 1
        rest = mask ^ lsb
        
        # Check if mask is a clique: rest is clique AND v adjacent to all in rest
        if valid[rest] and (rest & ~adj[v]) == 0:
            valid[mask] = True
        else:
            # Check if mask is an independent set: rest is indep AND v NOT adjacent to any in rest
            # We need a separate flag for independent sets
            pass
    
    # Need separate tracking for clique vs independent
    is_clique = [False] * (1 << n)
    is_indep = [False] * (1 << n)
    is_clique[0] = True
    is_indep[0] = True
    
    for mask in range(1, 1 << n):
        lsb = mask & (-mask)
        v = lsb.bit_length() - 1
        rest = mask ^ lsb
        
        # Clique: rest is clique AND v connected to all in rest
        if is_clique[rest] and (rest & ~adj[v]) == 0:
            is_clique[mask] = True
        
        # Independent: rest is independent AND v not connected to any in rest
        if is_indep[rest] and (rest & adj[v]) == 0:
            is_indep[mask] = True
        
        if is_clique[mask] or is_indep[mask]:
            valid[mask] = True
    
    # Now solve: minimum number of valid subsets that partition the full set
    # DP: dp[mask] = min number of parts to cover mask
    # dp[0] = 0
    # dp[mask] = 1 + min(dp[mask ^ sub] for sub in valid_subsets_of_mask)
    
    # But we need to enumerate valid subsets of each mask efficiently
    # For n=14, 2^14 = 16384 masks. For each mask, enumerate submasks.
    # Total work: sum over all masks of 2^(popcount(mask)) = 3^n = 3^14 ~ 4.8M
    # That's feasible.
    
    INF = float('inf')
    dp = [INF] * (1 << n)
    dp[0] = 0
    
    # Also track the best submask choice for reconstruction
    best_sub = [0] * (1 << n)
    
    for mask in range(1, 1 << n):
        if time.time() > deadline:
            break
        # Enumerate all submasks of mask
        sub = mask
        best = INF
        best_s = 0
        while sub > 0:
            if valid[sub]:
                val = dp[mask ^ sub] + 1
                if val < best:
                    best = val
                    best_s = sub
            sub = (sub - 1) & mask
        
        dp[mask] = best
        best_sub[mask] = best_s
    
    # Reconstruct the partition
    partition = []
    mask = full_mask
    while mask > 0:
        if time.time() > deadline:
            break
        sub = best_sub[mask]
        if sub == 0:
            # Fallback: just add remaining as individual vertices
            for v in range(n):
                if mask & (1 << v):
                    partition.append([v])
            break
        part = [v for v in range(n) if sub & (1 << v)]
        partition.append(part)
        mask ^= sub
    
    # Verify and return
    result = []
    for p in partition:
        result.append(list(p))
    
    # Ensure all vertices covered exactly once
    seen = [False] * n
    for p in result:
        for v in p:
            if seen[v]:
                # Should not happen
                pass
            seen[v] = True
    
    return result

# EVOLVE-BLOCK-END
