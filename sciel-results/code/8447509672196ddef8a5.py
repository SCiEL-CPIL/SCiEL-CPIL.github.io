"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import itertools
import numpy as np

P = {'n': 17, 'edges': [[0, 1], [0, 2], [0, 4], [0, 8], [0, 9], [0, 13], [0, 15], [0, 16], [1, 2], [1, 3], [1, 5], [1, 9], [1, 10], [1, 14], [1, 16], [2, 3], [2, 4], [2, 6], [2, 10], [2, 11], [2, 15], [3, 4], [3, 5], [3, 7], [3, 11], [3, 12], [3, 16], [4, 5], [4, 6], [4, 8], [4, 12], [4, 13], [5, 6], [5, 7], [5, 9], [5, 13], [5, 14], [6, 7], [6, 8], [6, 10], [6, 14], [6, 15], [7, 8], [7, 9], [7, 11], [7, 15], [7, 16], [8, 9], [8, 10], [8, 12], [8, 16], [9, 10], [9, 11], [9, 13], [10, 11], [10, 12], [10, 14], [11, 12], [11, 13], [11, 15], [12, 13], [12, 14], [12, 16], [13, 14], [13, 15], [14, 15], [14, 16], [15, 16]], 'homogeneous_cap': 4, 'minimum_vertices': 8}

# Precompute adjacency as bitmasks
N = P['n']
adj = [0] * N
for u, v in P['edges']:
    adj[u] |= (1 << v)
    adj[v] |= (1 << u)

def _distinct_degrees(subset_mask):
    """Compute the number of distinct degrees in the induced subgraph on subset_mask."""
    # Get the list of vertices in the subset
    verts = [i for i in range(N) if (subset_mask >> i) & 1]
    k = len(verts)
    if k < 2:
        return len(set([0] * k)) if k > 0 else 0
    
    # For each vertex in the subset, compute its degree within the induced subgraph
    degrees = []
    for v in verts:
        # Degree of v in induced subgraph = popcount(adj[v] & subset_mask)
        d = bin(adj[v] & subset_mask).count('1')
        degrees.append(d)
    
    return len(set(degrees))

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # Safety margin: 80s out of 90s limit
    
    best_count = 0
    best_subset = None
    
    # Search over all subsets of size 8 to 17
    # Total number of subsets: sum(C(17,k) for k=8..17) = 2^17 - sum(C(17,k) for k=0..7)
    # This is manageable: 2^17 = 131072
    # But we can be smarter: iterate by size
    
    # Precompute all vertex lists for each size
    for size in range(8, N + 1):
        if time.time() > deadline:
            break
        
        # Generate all combinations of 'size' vertices from 0..16
        for combo in itertools.combinations(range(N), size):
            if time.time() > deadline:
                break
            
            # Build bitmask
            mask = 0
            for v in combo:
                mask |= (1 << v)
            
            # Compute distinct degrees
            d = _distinct_degrees(mask)
            if d > best_count:
                best_count = d
                best_subset = combo
                # If we've found the maximum possible (size distinct degrees), we can't do better
                # But max distinct degrees is at most size (each vertex has a different degree)
                # Actually, max possible is min(size, N-1) but typically much less
                # We can't early-stop easily, but if best_count == size, that's the max for this size
                # But we want max across all sizes, so keep going unless we hit a global upper bound
    
    # The maximum possible distinct degrees in any induced subgraph of size k is at most k
    # And across all sizes, the max possible is 17 (but practically much less)
    # Since we're doing exhaustive search, we'll find the true maximum
    
    if best_subset is None:
        # Fallback: return first 8 vertices
        return list(range(8))
    
    return list(best_subset)

# EVOLVE-BLOCK-END
