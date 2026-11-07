import time

def construct():
    n = 14
    edges = P['edges']
    
    # Build adjacency matrix
    adj = [[0]*n for _ in range(n)]
    for u, v in edges:
        adj[u][v] = 1
        adj[v][u] = 1
        
    # Precompute valid masks (cliques or independent sets)
    # is_valid[mask] = True if mask is a clique or independent set
    is_valid = [False] * (1 << n)
    is_valid[0] = False # Empty set not allowed as a part
    
    # Helper to check if a mask is a clique
    def is_clique(mask):
        if mask == 0: return True
        # Get vertices
        verts = []
        m = mask
        while m:
            lsb = m & -m
            v = lsb.bit_length() - 1
            verts.append(v)
            m ^= lsb
        # Check all pairs
        for i in range(len(verts)):
            for j in range(i+1, len(verts)):
                if adj[verts[i]][verts[j]] == 0:
                    return False
        return True
        
    # Helper to check if a mask is an independent set
    def is_independent(mask):
        if mask == 0: return True
        verts = []
        m = mask
        while m:
            lsb = m & -m
            v = lsb.bit_length() - 1
            verts.append(v)
            m ^= lsb
        for i in range(len(verts)):
            for j in range(i+1, len(verts)):
                if adj[verts[i]][verts[j]] == 1:
                    return False
        return True

    # Precompute all valid masks
    valid_masks_by_vertex = [[] for _ in range(n)]
    for mask in range(1, 1 << n):
        if is_clique(mask) or is_independent(mask):
            is_valid[mask] = True
            # Add to lists for each vertex in mask
            m = mask
            while m:
                lsb = m & -m
                v = lsb.bit_length() - 1
                valid_masks_by_vertex[v].append(mask)
                m ^= lsb
                
    # Backtracking search
    # We want to find min k such that we can partition vertices into k valid parts.
    # can_cover(remaining, k)
    
    import functools
    @functools.lru_cache(maxsize=None)
    def can_cover(remaining, k):
        if k == 0:
            return remaining == 0
        if remaining == 0:
            return False # Parts must be non-empty, so we can't fill k slots with 0 vertices
            
        # Pruning: if remaining has more vertices than k * max_size? 
        # Max size of a valid part is n.
        # A better bound: The min number of parts to cover 'remaining' is at least 
        # the size of the largest clique in 'remaining' / (max clique size in a part)? No.
        
        # Pick the lowest set bit in 'remaining'
        lsb = remaining & -remaining
        v = lsb.bit_length() - 1
        
        # Try all valid masks M containing v and subset of remaining
        for M in valid_masks_by_vertex[v]:
            if (M & ~remaining) == 0:
                if can_cover(remaining ^ M, k - 1):
                    return True
        return False

    # Search for k
    start_time = time.time()
    deadline = start_time + 50 # 50s margin
    
    for k in range(1, n + 1):
        if time.time() > deadline:
            # Fallback to greedy if we run out of time (should not happen for n=14)
            break
        if can_cover((1 << n) - 1, k):
            # Found k. Now we need to reconstruct the partition.
            # The above function returns bool. We need the actual parts.
            # Let's modify to return the list of masks or just reconstruct.
            
            # Reconstruction
            parts = []
            remaining = (1 << n) - 1
            cur_k = k
            # We need a version of can_cover that records the path or we re-search with reconstruction.
            # Since n is small, we can just re-run with a list to store choices.
            
            def find_partition(remaining, k, path):
                if k == 0:
                    return remaining == 0
                if remaining == 0:
                    return False
                lsb = remaining & -remaining
                v = lsb.bit_length() - 1
                for M in valid_masks_by_vertex[v]:
                    if (M & ~remaining) == 0:
                        path.append(M)
                        if find_partition(remaining ^ M, k - 1, path):
                            return True
                        path.pop()
                return False
            
            path = []
            if find_partition((1 << n) - 1, k, path):
                # Convert masks to lists of vertices
                final_parts = []
                for mask in path:
                    verts = []
                    m = mask
                    while m:
                        lsb_m = m & -m
                        v_m = lsb_m.bit_length() - 1
                        verts.append(v_m)
                        m ^= lsb_m
                    final_parts.append(verts)
                return final_parts
            else:
                # Should not happen if can_cover returned True
                pass
    # Fallback greedy
    es = {tuple(e) for e in P['edges']}
    parts = []
    for v in range(P['n']):
        for part in parts:
            trial = part + [v]
            hits = [(u, w) in es for i, u in enumerate(trial) for w in trial[i+1:]]
            if all(hits) or not any(hits):
                part.append(v)
                break
        else:
            parts.append([v])
    return parts

P = {'n': 14, 'edges': [[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 10], [0, 12], [1, 4], [1, 9], [1, 11], [2, 3], [2, 5], [2, 6], [2, 7], [2, 12], [2, 13], [3, 4], [3, 5], [3, 8], [3, 11], [3, 12], [3, 13], [4, 5], [4, 7], [4, 13], [5, 8], [5, 9], [5, 10], [5, 12], [5, 13], [6, 8], [6, 11], [6, 13], [7, 12], [8, 9], [8, 13], [9, 11], [9, 12], [11, 13]]}
