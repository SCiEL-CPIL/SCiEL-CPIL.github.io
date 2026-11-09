"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

P = {'n': 14, 'edges': [[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 10], [0, 12], [1, 4], [1, 9], [1, 11], [2, 3], [2, 5], [2, 6], [2, 7], [2, 12], [2, 13], [3, 4], [3, 5], [3, 8], [3, 11], [3, 12], [3, 13], [4, 5], [4, 7], [4, 13], [5, 8], [5, 9], [5, 10], [5, 12], [5, 13], [6, 8], [6, 11], [6, 13], [7, 12], [8, 9], [8, 13], [9, 11], [9, 12], [11, 13]]}

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    
    n = P['n']
    edges = P['edges']
    adj = [0] * n
    for u, v in edges:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
    
    # Precompute valid subsets efficiently using DP
    valid = [False] * (1 << n)
    valid[0] = True
    
    # For each mask, check if it's a clique or independent set
    # A set is a clique if every pair is connected
    # A set is independent if no pair is connected
    # We can use the property: a set S is a clique iff for every v in S, S\{v} subset of N(v)
    # Or more simply, just check pairs but do it efficiently
    
    # Precompute popcount
    popcount = [0] * (1 << n)
    for i in range(1, 1 << n):
        popcount[i] = popcount[i >> 1] + (i & 1)
    
    # Efficiently check clique/independent using bit operations
    # For a mask to be a clique: for all u,v in mask with u<v, edge(u,v) exists
    # For a mask to be independent: for all u,v in mask with u<v, no edge(u,v)
    
    # Use DP approach: 
    # is_clique[mask] = is_clique[mask without lsb] AND (lsb connected to all in mask without lsb)
    # is_indep[mask] = is_indep[mask without lsb] AND (lsb not connected to any in mask without lsb)
    
    is_clique = [False] * (1 << n)
    is_indep = [False] * (1 << n)
    is_clique[0] = True
    is_indep[0] = True
    
    for mask in range(1, 1 << n):
        # Get lowest set bit
        lsb = mask & (-mask)
        v = lsb.bit_length() - 1
        rest = mask ^ lsb
        
        if rest == 0:
            is_clique[mask] = True
            is_indep[mask] = True
        else:
            # Check if v is connected to all vertices in rest (for clique)
            # v connected to all in rest means: rest & ~adj[v] == 0 (where ~adj[v] is non-neighbors)
            non_neighbors_v = ((1 << n) - 1) ^ adj[v] ^ (1 << v)
            if is_clique[rest] and (rest & non_neighbors_v) == 0:
                is_clique[mask] = True
            
            # Check if v is not connected to any vertex in rest (for independent)
            # v not connected to any in rest means: rest & adj[v] == 0
            if is_indep[rest] and (rest & adj[v]) == 0:
                is_indep[mask] = True
        
        valid[mask] = is_clique[mask] or is_indep[mask]
    
    valid_subs = [mask for mask in range(1, 1 << n) if valid[mask]]
    # Sort by size descending to find solutions faster
    valid_subs.sort(key=lambda x: popcount[x], reverse=True)
    
    max_valid_size = max((popcount[x] for x in valid_subs), default=1)
    
    from functools import lru_cache
    
    @lru_cache(None)
    def dfs(rem_mask, k):
        if time.time() > deadline:
            return False
        if k == 0:
            return rem_mask == 0
        if rem_mask == 0:
            return True
        if k == 1:
            return valid[rem_mask]
        # Lower bound: max number of vertices we can cover with k parts
        if popcount[rem_mask] > k * max_valid_size:
            return False
        
        # Additional pruning: if remaining vertices can't be covered by k valid subsets
        # Try to find a valid subset that contains the lowest set bit of rem_mask
        # This reduces symmetry
        lsb = rem_mask & (-rem_mask)
        
        for sub in valid_subs:
            if sub & rem_mask == sub:
                # Additional pruning: ensure sub contains lsb to reduce symmetry
                # But this might miss solutions, so let's be careful
                # Actually, we can require that the first part contains the lowest vertex
                # This is valid because we can always relabel parts
                if (sub & lsb) == 0:
                    continue
                if dfs(rem_mask ^ sub, k - 1):
                    return True
        return False

    full_mask = (1 << n) - 1
    best_k = n
    for k in range(1, n + 1):
        if dfs(full_mask, k):
            best_k = k
            break
        if time.time() > deadline:
            break
            
    @lru_cache(None)
    def get_parts(rem_mask, k):
        if k == 0 or rem_mask == 0:
            return []
        if k == 1:
            return [[i for i in range(n) if (rem_mask >> i) & 1]]
        
        lsb = rem_mask & (-rem_mask)
        
        for sub in valid_subs:
            if sub & rem_mask == sub:
                if (sub & lsb) == 0:
                    continue
                if dfs(rem_mask ^ sub, k - 1):
                    verts = [i for i in range(n) if (sub >> i) & 1]
                    rest = get_parts(rem_mask ^ sub, k - 1)
                    return [verts] + rest
        return []
        
    parts = get_parts(full_mask, best_k)
    # Ensure types are correct: each part is a list of ints
    return parts
# EVOLVE-BLOCK-END
