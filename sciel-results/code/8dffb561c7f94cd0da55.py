"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import itertools
import numpy as np

P = {'n': 17, 'edges': [[0, 1], [0, 2], [0, 4], [0, 8], [0, 9], [0, 13], [0, 15], [0, 16], [1, 2], [1, 3], [1, 5], [1, 9], [1, 10], [1, 14], [1, 16], [2, 3], [2, 4], [2, 6], [2, 10], [2, 11], [2, 15], [3, 4], [3, 5], [3, 7], [3, 11], [3, 12], [3, 16], [4, 5], [4, 6], [4, 8], [4, 12], [4, 13], [5, 6], [5, 7], [5, 9], [5, 13], [5, 14], [6, 7], [6, 8], [6, 10], [6, 14], [6, 15], [7, 8], [7, 9], [7, 11], [7, 15], [7, 16], [8, 9], [8, 10], [8, 12], [8, 16], [9, 10], [9, 11], [9, 13], [10, 11], [10, 12], [10, 14], [11, 12], [11, 13], [11, 15], [12, 13], [12, 14], [12, 16], [13, 14], [13, 15], [14, 15], [14, 16], [15, 16]], 'homogeneous_cap': 4, 'minimum_vertices': 8}

def construct():
    n = P['n']
    edges = P['edges']
    min_vertices = P['minimum_vertices']
    
    # Build adjacency matrix as numpy uint8
    adj = np.zeros((n, n), dtype=np.uint8)
    for u, v in edges:
        adj[u, v] = 1
        adj[v, u] = 1
    
    # Precompute row sums for quick degree lookups in full graph
    full_degrees = adj.sum(axis=1)
    
    deadline = time.time() + 85  # 85s safety margin within 90s limit
    best_score = 0
    best_subset = list(range(min_vertices))
    
    # Precompute bitmask for each vertex's neighbors
    neighbor_masks = [0] * n
    for u, v in edges:
        neighbor_masks[u] |= (1 << v)
        neighbor_masks[v] |= (1 << u)
    
    # Build a lookup table: for each possible subset mask (up to 17 bits),
    # we can't precompute all 2^17, but we can use popcount on masked adjacency
    
    def compute_distinct_degrees(subset_tuple):
        """Compute number of distinct degrees in induced subgraph."""
        mask = 0
        for v in subset_tuple:
            mask |= (1 << v)
        degs = []
        for v in subset_tuple:
            d = bin(neighbor_masks[v] & mask).count('1')
            degs.append(d)
        return len(set(degs))
    
    # Search sizes from 8 to 17
    for size in range(8, n + 1):
        if time.time() > deadline:
            break
        # Pruning: max possible distinct degrees is size (each vertex unique degree)
        # but degrees range from 0 to size-1, so max distinct is size
        # If current best is already >= size, no need to search this size
        # Actually max distinct degrees in a graph of size s is at most s
        # (degrees are 0..s-1, so at most s distinct values)
        # But we can only beat best_score if size > best_score
        if size <= best_score:
            continue
        
        for subset in itertools.combinations(range(n), size):
            if time.time() > deadline:
                break
            distinct = compute_distinct_degrees(subset)
            if distinct > best_score:
                best_score = distinct
                best_subset = list(subset)
    
    # Ensure we return at least min_vertices
    if len(best_subset) < min_vertices:
        best_subset = list(range(min_vertices))
    
    return list(best_subset)
# EVOLVE-BLOCK-END
