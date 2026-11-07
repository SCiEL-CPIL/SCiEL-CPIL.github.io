"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

P = {'n': 17, 'edges': [[0, 1], [0, 2], [0, 4], [0, 8], [0, 9], [0, 13], [0, 15], [0, 16], [1, 2], [1, 3], [1, 5], [1, 9], [1, 10], [1, 14], [1, 16], [2, 3], [2, 4], [2, 6], [2, 10], [2, 11], [2, 15], [3, 4], [3, 5], [3, 7], [3, 11], [3, 12], [3, 16], [4, 5], [4, 6], [4, 8], [4, 12], [4, 13], [5, 6], [5, 7], [5, 9], [5, 13], [5, 14], [6, 7], [6, 8], [6, 10], [6, 14], [6, 15], [7, 8], [7, 9], [7, 11], [7, 15], [7, 16], [8, 9], [8, 10], [8, 12], [8, 16], [9, 10], [9, 11], [9, 13], [10, 11], [10, 12], [10, 14], [11, 12], [11, 13], [11, 15], [12, 13], [12, 14], [12, 16], [13, 14], [13, 15], [14, 15], [14, 16], [15, 16]], 'homogeneous_cap': 4, 'minimum_vertices': 8}

def construct():
    deadline = time.time() + 55  # Safety margin well before 60s limit
    
    n = P['n']
    edges = P['edges']
    min_v = P['minimum_vertices']
    
    # Build adjacency bitmask
    adj_mask = [0] * n
    for u, v in edges:
        adj_mask[u] |= (1 << v)
        adj_mask[v] |= (1 << u)
    
    total_masks = 1 << n
    
    # Precompute degree tuple for every mask:
    # deg_tuple[mask] = tuple of (induced degree of vertex v in mask) for v in 0..n-1
    # Only vertices in the mask have meaningful degrees; others get -1 (ignored)
    # We store as a tuple of small ints for fast len(set(...))
    deg_tuple = [None] * total_masks
    deg_tuple[0] = tuple()
    
    for mask in range(1, total_masks):
        lsb = mask & (-mask)
        v = lsb.bit_length() - 1
        prev = mask ^ lsb
        # Get the degree of vertex v within 'mask'
        # v's neighbors in mask: adj_mask[v] & mask
        # But we need to count how many of v's neighbors are in mask
        # Since v is in mask, we count neighbors of v that are also in mask
        deg_v = 0
        am = adj_mask[v]
        m = mask
        while m:
            b = m & (-m)
            idx = b.bit_length() - 1
            if (am >> idx) & 1:
                deg_v += 1
            m ^= b
        # Build tuple: for each vertex in mask, store its degree
        # We need to reconstruct the full tuple from prev's tuple plus v's degree
        # prev_tuple has entries for all vertices in prev (mask without v)
        # We need to insert deg_v at position v
        prev_tup = deg_tuple[prev]
        # Build new tuple by inserting deg_v at position v
        # Since prev_tup contains degrees for vertices in prev (not including v),
        # and we want a tuple indexed by vertex position:
        # Actually, let's store it differently: store a tuple where index = vertex,
        # and value = degree if vertex is in mask, else -1
        # This makes it O(n) to build but O(1) to compute distinct count
        # For n=17, this is fine
        
        # Rebuild: create a list of length n, fill from prev_tup, set v's degree
        # Actually for speed, let's just store the tuple of degrees for vertices IN the mask
        # in sorted vertex order, and compute distinct from that
        # But that requires knowing which vertices are in the mask...
        
        # Simpler approach: store a fixed-length tuple of length n
        # where entry i = degree of vertex i in mask if i is in mask, else -1
        new_tup = list(prev_tup) if prev_tup else [-1] * n
        new_tup[v] = deg_v
        deg_tuple[mask] = tuple(new_tup)
    
    # Group masks by popcount (size)
    # masks_by_size[s] = list of masks with exactly s bits set
    masks_by_size = [[] for _ in range(n + 1)]
    for mask in range(1, total_masks):
        # Count bits
        c = 0
        m = mask
        while m:
            c += 1
            m &= m - 1
        masks_by_size[c].append(mask)
    
    best_distinct = -1
    best_mask = 0
    
    # Search from largest size to smallest
    for size in range(n, min_v - 1, -1):
        # Prune: can't get more than 'size' distinct degrees
        if size <= best_distinct:
            continue
        
        for mask in masks_by_size[size]:
            if time.time() > deadline:
                break
            
            # Compute distinct degrees: count distinct values in deg_tuple[mask]
            # that are >= 0 (i.e., vertices that are in the mask)
            tup = deg_tuple[mask]
            # Use a small set; degrees are in range [0, size-1]
            seen = set()
            for d in tup:
                if d >= 0:
                    seen.add(d)
            distinct = len(seen)
            
            if distinct > best_distinct:
                best_distinct = distinct
                best_mask = mask
                if best_distinct == size:
                    # Can't do better for this size
                    break
        
        if time.time() > deadline:
            break
    
    # Convert best_mask to list of vertex indices
    result = []
    m = best_mask
    while m:
        b = m & (-m)
        result.append(b.bit_length() - 1)
        m ^= b
    
    # Ensure at least min_v vertices (should be by construction)
    if len(result) < min_v:
        result = list(range(min_v))
    
    return [int(x) for x in result]

# EVOLVE-BLOCK-END
