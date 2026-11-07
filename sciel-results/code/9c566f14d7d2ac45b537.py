"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

P = {'n': 17, 'edges': [[0, 1], [0, 2], [0, 4], [0, 8], [0, 9], [0, 13], [0, 15], [0, 16], [1, 2], [1, 3], [1, 5], [1, 9], [1, 10], [1, 14], [1, 16], [2, 3], [2, 4], [2, 6], [2, 10], [2, 11], [2, 15], [3, 4], [3, 5], [3, 7], [3, 11], [3, 12], [3, 16], [4, 5], [4, 6], [4, 8], [4, 12], [4, 13], [5, 6], [5, 7], [5, 9], [5, 13], [5, 14], [6, 7], [6, 8], [6, 10], [6, 14], [6, 15], [7, 8], [7, 9], [7, 11], [7, 15], [7, 16], [8, 9], [8, 10], [8, 12], [8, 16], [9, 10], [9, 11], [9, 13], [10, 11], [10, 12], [10, 14], [11, 12], [11, 13], [11, 15], [12, 13], [12, 14], [12, 16], [13, 14], [13, 15], [14, 15], [14, 16], [15, 16]], 'homogeneous_cap': 4, 'minimum_vertices': 8}

def construct():
    n = P['n']
    edges = P['edges']
    min_size = P['minimum_vertices']
    
    # Build adjacency bitmasks
    adj = [0] * n
    for u, v in edges:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        
    deadline = time.monotonic() + 80.0  # Safety margin
    
    # Precompute vertex lists for all masks of size >= min_size
    # This avoids repeated bit extraction
    best_deg = 0
    best_vertices = None
    
    # Group masks by their popcount for better organization
    masks_by_size = [[] for _ in range(n + 1)]
    for mask in range(1 << n):
        pc = mask.bit_count()
        if pc >= min_size:
            masks_by_size[pc].append(mask)
    
    # Process from largest subsets first (more room for distinct degrees)
    for size in range(n, min_size - 1, -1):
        if time.monotonic() > deadline:
            break
        
        for mask in masks_by_size[size]:
            if time.monotonic() > deadline:
                break
            
            # Compute distinct degrees
            degs = set()
            m = mask
            while m:
                low = m & (-m)
                v = low.bit_length() - 1
                d = (adj[v] & mask).bit_count()
                degs.add(d)
                m ^= low
            
            nd = len(degs)
            if nd > best_deg:
                best_deg = nd
                # Extract vertices for this mask
                verts = []
                m2 = mask
                while m2:
                    low = m2 & (-m2)
                    v = low.bit_length() - 1
                    verts.append(v)
                    m2 ^= low
                best_vertices = verts
                
                # If we've achieved the theoretical maximum for this size, 
                # and larger sizes can't beat it, we can stop
                # Max distinct degrees for size k is at most k (degrees 0..k-1)
                # But actually max is min(k, 16) since degree <= k-1
                if nd == size:
                    # Perfect: all degrees distinct. Can't do better for this size.
                    # Larger sizes might have more, but let's check if worth continuing
                    pass
    
    if best_vertices is None:
        # Fallback, though min_size=8 ensures we find something
        best_vertices = list(range(min_size))
        
    return [int(v) for v in best_vertices]

# EVOLVE-BLOCK-END
