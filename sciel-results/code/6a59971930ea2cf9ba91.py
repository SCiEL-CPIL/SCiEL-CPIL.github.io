"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import itertools

P = {'n': 17, 'edges': [[0, 1], [0, 2], [0, 4], [0, 8], [0, 9], [0, 13], [0, 15], [0, 16], [1, 2], [1, 3], [1, 5], [1, 9], [1, 10], [1, 14], [1, 16], [2, 3], [2, 4], [2, 6], [2, 10], [2, 11], [2, 15], [3, 4], [3, 5], [3, 7], [3, 11], [3, 12], [3, 16], [4, 5], [4, 6], [4, 8], [4, 12], [4, 13], [5, 6], [5, 7], [5, 9], [5, 13], [5, 14], [6, 7], [6, 8], [6, 10], [6, 14], [6, 15], [7, 8], [7, 9], [7, 11], [7, 15], [7, 16], [8, 9], [8, 10], [8, 12], [8, 16], [9, 10], [9, 11], [9, 13], [10, 11], [10, 12], [10, 14], [11, 12], [11, 13], [11, 15], [12, 13], [12, 14], [12, 16], [13, 14], [13, 15], [14, 15], [14, 16], [15, 16]], 'homogeneous_cap': 4, 'minimum_vertices': 8}

def construct():
    n = P['n']
    min_vertices = P['minimum_vertices']
    
    # Build adjacency bitmask
    adj = [0] * n
    for u, v in P['edges']:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        
    best_score = -1
    best_subset = None
    best_mask = 0
    
    # Precompute popcount for all masks up to 2^17? 
    # Or just use bin(x).count('1') which is fast in Py3
    # Let's iterate all masks
    total_masks = 1 << n
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 50.0 # 50s margin
    
    for mask in range(total_masks):
        if time.time() > deadline:
            break
            
        # Check size
        size = bin(mask).count('1')
        if size < min_vertices:
            continue
            
        # Calculate distinct degrees
        degrees = set()
        # Iterate through vertices in the mask
        m = mask
        while m:
            # Get lowest set bit
            lsb = m & -m
            v = lsb.bit_length() - 1
            # Calculate degree of v in this mask
            # Neighbors of v in the mask
            deg = bin(adj[v] & mask).count('1')
            degrees.add(deg)
            m ^= lsb
            
        score = len(degrees)
        if score > best_score:
            best_score = score
            best_mask = mask
            # Extract vertices for best_subset
            verts = []
            mm = mask
            while mm:
                lsb = mm & -mm
                v = lsb.bit_length() - 1
                verts.append(v)
                mm ^= lsb
            best_subset = verts
            
            # Early exit if we hit the theoretical max?
            # Max distinct degrees for size k is k.
            # If size is 8, max is 8. If we find 8, we can stop? 
            # But larger sets might have more distinct degrees? No, max distinct degrees is bounded by size.
            # However, a set of size 10 can have 10 distinct degrees.
            # So we can't just stop at 8 unless we know no larger set can beat it.
            # But max possible score is 17 (if all 17 vertices have distinct degrees).
            # So we should keep searching.
            
    # Return the best subset found
    if best_subset is None:
        # Fallback, should not happen
        return list(range(min_vertices))
        
    return best_subset

# EVOLVE-BLOCK-END
