import time

P = {'n': 14, 'edges': [[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 10], [0, 12], [1, 4], [1, 9], [1, 11], [2, 3], [2, 5], [2, 6], [2, 7], [2, 12], [2, 13], [3, 4], [3, 5], [3, 8], [3, 11], [3, 12], [3, 13], [4, 5], [4, 7], [4, 13], [5, 8], [5, 9], [5, 10], [5, 12], [5, 13], [6, 8], [6, 11], [6, 13], [7, 12], [8, 9], [8, 13], [9, 11], [9, 12], [11, 13]]}

def construct():
    n = P['n']
    edges = P['edges']
    
    # Precompute edge set for fast lookup
    edge_set = set()
    for u, v in edges:
        edge_set.add((u, v))
        edge_set.add((v, u))
        
    # Precompute valid masks
    # valid[mask] is True if mask is a clique or an independent set
    # We represent mask as an integer
    size = 1 << n
    is_clique = [False] * size
    is_independent = [False] * size
    
    is_clique[0] = True
    is_independent[0] = True
    
    for mask in range(1, size):
        # Find lowest set bit
        v = (mask & -mask).bit_length() - 1
        rest = mask ^ (1 << v)
        
        # Check clique: v must be connected to all in rest, and rest must be clique
        # Optimization: check connectivity first
        is_clique_candidate = is_clique[rest]
        if is_clique_candidate:
            # Check if v is connected to all vertices in rest
            ok = True
            r = rest
            while r:
                u = (r & -r).bit_length() - 1
                if (v, u) not in edge_set:
                    ok = False
                    break
                r ^= (1 << u)
            if ok:
                is_clique[mask] = True
                
        # Check independent: v must NOT be connected to any in rest, and rest must be independent
        is_independent_candidate = is_independent[rest]
        if is_independent_candidate:
            # Check if v is NOT connected to any vertices in rest
            ok = True
            r = rest
            while r:
                u = (r & -r).bit_length() - 1
                if (v, u) in edge_set:
                    ok = False
                    break
                r ^= (1 << u)
            if ok:
                is_independent[mask] = True
                
    valid = [is_clique[m] or is_independent[m] for m in range(size)]
    
    full_mask = (1 << n) - 1
    
    # Backtracking search
    # parts is a list of masks representing the current parts
    def search(unassigned, parts, k_limit):
        if unassigned == 0:
            return parts
        
        if len(parts) >= k_limit:
            return None
            
        # Pick the lowest unassigned vertex
        v_bit = unassigned & -unassigned
        v = v_bit.bit_length() - 1
        
        # Option 1: Add v to an existing part
        for i in range(len(parts)):
            p_mask = parts[i]
            new_p_mask = p_mask | v_bit
            if valid[new_p_mask]:
                # Create new parts list
                new_parts = parts.copy()
                new_parts[i] = new_p_mask
                res = search(unassigned ^ v_bit, new_parts, k_limit)
                if res is not None:
                    return res
                    
        # Option 2: Start a new part with v
        # We must ensure we don't exceed k_limit
        if len(parts) < k_limit:
            new_parts = parts + [v_bit]
            res = search(unassigned ^ v_bit, new_parts, k_limit)
            if res is not None:
                return res
                
        return None

    # Try k from 1 to n
    for k in range(1, n + 1):
        res_masks = search(full_mask, [], k)
        if res_masks is not None:
            # Convert masks back to lists of vertices
            partition = []
            for m in res_masks:
                verts = []
                for v in range(n):
                    if (m >> v) & 1:
                        verts.append(v)
                partition.append(verts)
            # Ensure types are correct (list of lists of ints)
            return partition
            
    # Fallback (should not happen)
    return [[v] for v in range(n)]
