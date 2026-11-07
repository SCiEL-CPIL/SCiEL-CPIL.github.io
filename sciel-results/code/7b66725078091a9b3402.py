"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

P = {'n': 17, 'edges': [[0, 1], [0, 2], [0, 4], [0, 8], [0, 9], [0, 13], [0, 15], [0, 16], [1, 2], [1, 3], [1, 5], [1, 9], [1, 10], [1, 14], [1, 16], [2, 3], [2, 4], [2, 6], [2, 10], [2, 11], [2, 15], [3, 4], [3, 5], [3, 7], [3, 11], [3, 12], [3, 16], [4, 5], [4, 6], [4, 8], [4, 12], [4, 13], [5, 6], [5, 7], [5, 9], [5, 13], [5, 14], [6, 7], [6, 8], [6, 10], [6, 14], [6, 15], [7, 8], [7, 9], [7, 11], [7, 15], [7, 16], [8, 9], [8, 10], [8, 12], [8, 16], [9, 10], [9, 11], [9, 13], [10, 11], [10, 12], [10, 14], [11, 12], [11, 13], [11, 15], [12, 13], [12, 14], [12, 16], [13, 14], [13, 15], [14, 15], [14, 16], [15, 16]], 'homogeneous_cap': 4, 'minimum_vertices': 8}

def construct():
    n = P['n']
    edges = P['edges']
    min_vertices = P['minimum_vertices']
    
    # Build adjacency masks
    # adj_mask[i] is an integer where bit j is 1 if (i,j) is an edge
    adj_mask = [0] * n
    for u, v in edges:
        adj_mask[u] |= (1 << v)
        adj_mask[v] |= (1 << u)
        
    # Precompute popcount for all possible masks? 
    # 2^17 = 131,072. This is small.
    # But we only need popcount of (adj_mask[v] & mask).
    # Python int.bit_count() is fast.
    
    best_count = -1
    best_subset = None
    
    # Deadline
    deadline = time.time() + 55.0
    
    # Iterate over all masks with at least min_vertices bits set
    # Total masks 2^17.
    # We can iterate 0 to 2^17 - 1.
    
    # Optimization: 
    # If we find a count == n (17), we can stop immediately.
    # Max distinct degrees in subset of size k is k.
    # So max possible is 17.
    
    # We can iterate by popcount of the mask to ensure we check sizes >= 8.
    # Or just iterate all masks and check bit_count >= 8.
    
    for mask in range(1 << n):
        if time.time() > deadline:
            if best_subset is not None:
                # Convert mask to list of vertices
                subset = [i for i in range(n) if (mask >> i) & 1]
                return subset
            else:
                return list(range(min_vertices))
                
        k = mask.bit_count()
        if k < min_vertices:
            continue
            
        # Calculate degrees for vertices in mask
        # We only care about vertices IN the mask.
        # deg[v] = bit_count(adj_mask[v] & mask)
        
        # We need distinct degrees.
        # Use a set to track distinct degrees.
        # Early termination: if len(set) + (k - processed) <= best_count, skip?
        # k is small (<=17), so just compute.
        
        degrees = set()
        # Iterate bits in mask
        m = mask
        while m:
            # Get lowest set bit
            lsb = m & -m
            v = lsb.bit_length() - 1
            m ^= lsb
            
            deg = (adj_mask[v] & mask).bit_count()
            degrees.add(deg)
            
            # Optional pruning: if we have seen all possible degrees 0..k-1?
            # Max distinct is k. If len(degrees) == k, we can break inner loop?
            # No, we need to process all to ensure we don't miss something? 
            # If we have k distinct degrees, that's the max possible for this k.
            # We can break.
            if len(degrees) == k:
                break
                
        distinct_count = len(degrees)
        
        if distinct_count > best_count:
            best_count = distinct_count
            best_subset = mask
            
            # If we hit the theoretical max for this k, and k is large, 
            # we might be able to prune other k's?
            # Max distinct for k is k.
            # If best_count == k, we can't beat it with same k.
            # But we can beat it with larger k.
            # So we continue.
            
    if best_subset is not None:
        subset = [i for i in range(n) if (best_subset >> i) & 1]
        return subset
    else:
        return list(range(min_vertices))
# EVOLVE-BLOCK-END
