import time
import itertools

# EVOLVE-BLOCK-START
# Graph definition provided in the task
P = {'n': 17, 'edges': [[0, 1], [0, 2], [0, 4], [0, 8], [0, 9], [0, 13], [0, 15], [0, 16], [1, 2], [1, 3], [1, 5], [1, 9], [1, 10], [1, 14], [1, 16], [2, 3], [2, 4], [2, 6], [2, 10], [2, 11], [2, 15], [3, 4], [3, 5], [3, 7], [3, 11], [3, 12], [3, 16], [4, 5], [4, 6], [4, 8], [4, 12], [4, 13], [5, 6], [5, 7], [5, 9], [5, 13], [5, 14], [6, 7], [6, 8], [6, 10], [6, 14], [6, 15], [7, 8], [7, 9], [7, 11], [7, 15], [7, 16], [8, 9], [8, 10], [8, 12], [8, 16], [9, 10], [9, 11], [9, 13], [10, 11], [10, 12], [10, 14], [11, 12], [11, 13], [11, 15], [12, 13], [12, 14], [12, 16], [13, 14], [13, 15], [14, 15], [14, 16], [15, 16]], 'homogeneous_cap': 4, 'minimum_vertices': 8}


def construct():
    """
    Finds a subset of at least 8 vertices that maximizes the number of distinct degrees in the induced subgraph.
    Returns a list of vertex indices.
    """
    n = P['n']
    edges = P['edges']
    min_vertices = P['minimum_vertices']
    
    # Build adjacency bitmasks
    adj = [0] * n
    for a, b in edges:
        adj[a] |= (1 << b)
        adj[b] |= (1 << a)
        
    best_distinct_count = -1
    best_mask = 0
    best_vertices = []
    
    # Precompute vertex indices for each mask? 
    # Or just iterate masks and check popcount.
    # Total masks is 2^17 = 131072. This is small.
    
    for mask in range(1 << n):
        # Check if mask has at least min_vertices bits set
        # bit_count is available in Python 3.8+
        size = mask.bit_count()
        if size < min_vertices:
            continue
            
        # Calculate distinct degrees in the induced subgraph
        # Degree of v in mask is popcount(adj[v] & mask)
        degrees = set()
        m = mask
        while m:
            lsb = m & -m
            v = lsb.bit_length() - 1
            # Get neighbors of v within the mask
            deg = (adj[v] & mask).bit_count()
            degrees.add(deg)
            m ^= lsb
            
        distinct_count = len(degrees)
        
        if distinct_count > best_distinct_count:
            best_distinct_count = distinct_count
            best_mask = mask
            # Optional: Early exit if we found a perfect set of distinct degrees?
            # Max possible distinct degrees is size (if all degrees are different).
            # But degrees are bounded by size-1. 
            # If distinct_count == size, we have a "rainbow" degree sequence.
            # However, it's rare. We can just finish the loop as it's fast.
            
    # Extract vertices from best_mask
    best_vertices = []
    m = best_mask
    while m:
        lsb = m & -m
        v = lsb.bit_length() - 1
        best_vertices.append(int(v))
        m ^= lsb
        
    # Ensure we return a list of ints
    return best_vertices
# EVOLVE-BLOCK-END
