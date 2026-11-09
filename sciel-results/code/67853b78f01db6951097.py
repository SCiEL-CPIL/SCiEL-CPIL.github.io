"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

P = {'n': 14, 'edges': [[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 10], [0, 12], [1, 4], [1, 9], [1, 11], [2, 3], [2, 5], [2, 6], [2, 7], [2, 12], [2, 13], [3, 4], [3, 5], [3, 8], [3, 11], [3, 12], [3, 13], [4, 5], [4, 7], [4, 13], [5, 8], [5, 9], [5, 10], [5, 12], [5, 13], [6, 8], [6, 11], [6, 13], [7, 12], [8, 9], [8, 13], [9, 11], [9, 12], [11, 13]]}

def construct():
    start_time = time.time()
    deadline = start_time + 50.0
    
    n = P['n']
    edges = P['edges']
    adj = [0] * n
    for u, v in edges:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
    
    # Precompute valid subsets
    valid = [False] * (1 << n)
    valid[0] = True
    for mask in range(1, 1 << n):
        is_clique = True
        is_independent = True
        verts = [i for i in range(n) if (mask >> i) & 1]
        for i in range(len(verts)):
            for j in range(i+1, len(verts)):
                u, v = verts[i], verts[j]
                if (adj[u] >> v) & 1:
                    is_independent = False
                else:
                    is_clique = False
                if not is_clique and not is_independent:
                    break
            if not is_clique and not is_independent:
                break
        valid[mask] = is_clique or is_independent
        
    valid_subs = [mask for mask in range(1, 1 << n) if valid[mask]]
    # Sort by size descending to find solutions faster
    valid_subs.sort(key=lambda x: bin(x).count('1'), reverse=True)
    
    # Precompute popcount
    popcount = [0] * (1 << n)
    for i in range(1, 1 << n):
        popcount[i] = popcount[i >> 1] + (i & 1)
        
    max_valid_size = max((bin(x).count('1') for x in valid_subs), default=1)
    
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
            
        for sub in valid_subs:
            if sub & rem_mask == sub:
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
        
        for sub in valid_subs:
            if sub & rem_mask == sub:
                if dfs(rem_mask ^ sub, k - 1):
                    verts = [i for i in range(n) if (sub >> i) & 1]
                    rest = get_parts(rem_mask ^ sub, k - 1)
                    return [verts] + rest
        return []
        
    parts = get_parts(full_mask, best_k)
    # Ensure types are correct
    return parts
# EVOLVE-BLOCK-END
