# EVOLVE-BLOCK-START
P = {'n': 17, 'edges': [[0, 1], [0, 2], [0, 4], [0, 8], [0, 9], [0, 13], [0, 15], [0, 16], [1, 2], [1, 3], [1, 5], [1, 9], [1, 10], [1, 14], [1, 16], [2, 3], [2, 4], [2, 6], [2, 10], [2, 11], [2, 15], [3, 4], [3, 5], [3, 7], [3, 11], [3, 12], [3, 16], [4, 5], [4, 6], [4, 8], [4, 12], [4, 13], [5, 6], [5, 7], [5, 9], [5, 13], [5, 14], [6, 7], [6, 8], [6, 10], [6, 14], [6, 15], [7, 8], [7, 9], [7, 11], [7, 15], [7, 16], [8, 9], [8, 10], [8, 12], [8, 16], [9, 10], [9, 11], [9, 13], [10, 11], [10, 12], [10, 14], [11, 12], [11, 13], [11, 15], [12, 13], [12, 14], [12, 16], [13, 14], [13, 15], [14, 15], [14, 16], [15, 16]], 'homogeneous_cap': 4, 'minimum_vertices': 8}

def construct():
    n = P['n']
    edges = P['edges']
    min_verts = P['minimum_vertices']
    
    # Build adjacency bitmasks
    adj_mask = [0] * n
    for u, v in edges:
        adj_mask[u] |= (1 << v)
        adj_mask[v] |= (1 << u)
    
    best_subset_mask = 0
    best_distinct = -1
    
    # Iterate over all possible subsets
    # We only care about subsets with size >= min_verts
    # There are 2^17 = 131072 subsets
    for mask in range(1 << n):
        # Quick check on size
        size = mask.bit_count()
        if size < min_verts:
            continue
            
        # Calculate degrees for each vertex in the subset
        degrees = set()
        # Iterate through vertices in the subset
        m = mask
        while m:
            # Get lowest set bit
            lsb = m & (-m)
            v = lsb.bit_length() - 1
            m ^= lsb
            
            # Degree of v in induced subgraph is number of neighbors in mask
            deg = (mask & adj_mask[v]).bit_count()
            degrees.add(deg)
            
            # Early break if we already have 'size' distinct degrees? 
            # Max distinct is size. If we have size distinct, we can't do better for this subset.
            # But we are checking distinctness for THIS subset.
            # If len(degrees) == size, it's the max possible for this size.
            # We can break early from degree calculation? No, we need the set.
            # But if we break, we know distinct = size.
            if len(degrees) == size:
                # We have all distinct degrees. This is the max possible for this size.
                # We can stop checking other vertices for this mask? 
                # Yes, distinct count cannot exceed size.
                # So distinct = size.
                break
        
        distinct = len(degrees)
        
        if distinct > best_distinct:
            best_distinct = distinct
            best_subset_mask = mask
            
            # If we found a subset of size 'size' with 'size' distinct degrees,
            # can we do better?
            # Max distinct for size k is k.
            # If we find a subset of size 17 with 17 distinct, that's global max 17.
            # If we find size 16 with 16 distinct, maybe size 17 has 17?
            # We must continue to check larger sizes.
            # However, if best_distinct == 17 (max possible for n=17), we can stop entirely.
            if best_distinct == n:
                break

    # Convert mask to list of vertices
    result = []
    m = best_subset_mask
    while m:
        lsb = m & (-m)
        v = lsb.bit_length() - 1
        result.append(v)
        m ^= lsb
        
    return result
# EVOLVE-BLOCK-END
