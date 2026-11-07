"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

P = {'n': 14, 'edges': [[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 10], [0, 12], [1, 4], [1, 9], [1, 11], [2, 3], [2, 5], [2, 6], [2, 7], [2, 12], [2, 13], [3, 4], [3, 5], [3, 8], [3, 11], [3, 12], [3, 13], [4, 5], [4, 7], [4, 13], [5, 8], [5, 9], [5, 10], [5, 12], [5, 13], [6, 8], [6, 11], [6, 13], [7, 12], [8, 9], [8, 13], [9, 11], [9, 12], [11, 13]]}

def construct():
    start_time = time.time()
    deadline = start_time + 55  # 55s safety margin before 60s limit
    n = P['n']
    edges = P['edges']
    
    # Build adjacency bitmasks
    adj = [0] * n
    for u, v in edges:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        
    full_mask = (1 << n) - 1
    num_subsets = 1 << n
    
    # Precompute validity of each subset: is it a clique or independent set?
    # Use a more efficient approach: for each subset, check if it's a clique or ind set
    # A subset S is a clique iff for every vertex v in S, all other vertices in S are adjacent to v
    # A subset S is an independent set iff for every vertex v in S, no other vertex in S is adjacent to v
    
    # Efficient method: 
    # - S is a clique iff S == union of {v} U N(v) for all v in S, i.e., S & ~N[v] == 0 for all v in S where N[v] includes v
    # Actually simpler: S is a clique iff for all v in S, (S & ~adj[v] & ~(1<<v)) == 0
    # S is an independent set iff for all v in S, (S & adj[v]) == 0
    
    # Precompute closed neighborhoods N[v] = adj[v] | (1 << v)
    closed_neigh = [adj[v] | (1 << v) for v in range(n)]
    
    is_valid = [False] * num_subsets
    is_valid[0] = False
    
    for S in range(1, num_subsets):
        # Check clique: for all v in S, S must be subset of closed_neigh[v]
        # i.e., S & ~closed_neigh[v] == 0 for all v in S
        is_clique = True
        is_ind = True
        temp = S
        v = 0
        while temp:
            if temp & 1:
                if S & ~closed_neigh[v]:
                    is_clique = False
                if S & adj[v]:
                    is_ind = False
                if not is_clique and not is_ind:
                    break
            temp >>= 1
            v += 1
        
        if is_clique or is_ind:
            is_valid[S] = True
    
    # Check deadline
    if time.time() > deadline:
        # Fallback: return individual vertices as parts (always valid)
        return [[i] for i in range(n)]
    
    # DP: dp[S] = min number of parts to partition S
    INF = 10**9
    dp = [INF] * num_subsets
    parent_part = [0] * num_subsets
    dp[0] = 0
    
    # Precompute list of valid submasks for each S to speed up DP
    # For each S, store all valid submasks of S
    # This is expensive to precompute, so we'll just iterate submasks on the fly
    
    for S in range(1, num_subsets):
        if is_valid[S]:
            dp[S] = 1
            parent_part[S] = S
            continue
        
        # Find best valid submask
        best_val = INF
        best_sub = 0
        sub = (S - 1) & S
        while sub:
            if is_valid[sub]:
                rest = S ^ sub
                val = dp[rest] + 1
                if val < best_val:
                    best_val = val
                    best_sub = sub
                    if best_val == 2:  # Can't do better than 2 if S itself isn't valid
                        break
            sub = (sub - 1) & S
            
            if time.time() > deadline:
                break
        
        if best_sub != 0:
            dp[S] = best_val
            parent_part[S] = best_sub
        else:
            # Fallback: use single vertex (should not happen)
            # Find lowest set bit
            lowbit = S & (-S)
            dp[S] = dp[S ^ lowbit] + 1
            parent_part[S] = lowbit
    
    # Reconstruct the partition
    parts = []
    S = full_mask
    while S:
        part_mask = parent_part[S]
        if part_mask == 0:
            # Should not happen, but safety
            lowbit = S & (-S)
            part_mask = lowbit
        part_verts = [i for i in range(n) if (part_mask >> i) & 1]
        parts.append(part_verts)
        S = S ^ part_mask
    
    # Ensure all vertices are covered exactly once
    # Verify and fix if needed
    seen = [False] * n
    for part in parts:
        for v in part:
            if not (0 <= v < n) or seen[v]:
                # Invalid, rebuild
                return [[i] for i in range(n)]
            seen[v] = True
    
    if not all(seen):
        return [[i] for i in range(n)]
    
    # Convert to required types
    final_parts = []
    for part in parts:
        final_parts.append([int(v) for v in part])
        
    return final_parts

# EVOLVE-BLOCK-END
