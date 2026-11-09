"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import itertools

P = {'n': 17, 'edges': [[0, 1], [0, 2], [0, 4], [0, 8], [0, 9], [0, 13], [0, 15], [0, 16], [1, 2], [1, 3], [1, 5], [1, 9], [1, 10], [1, 14], [1, 16], [2, 3], [2, 4], [2, 6], [2, 10], [2, 11], [2, 15], [3, 4], [3, 5], [3, 7], [3, 11], [3, 12], [3, 16], [4, 5], [4, 6], [4, 8], [4, 12], [4, 13], [5, 6], [5, 7], [5, 9], [5, 13], [5, 14], [6, 7], [6, 8], [6, 10], [6, 14], [6, 15], [7, 8], [7, 9], [7, 11], [7, 15], [7, 16], [8, 9], [8, 10], [8, 12], [8, 16], [9, 10], [9, 11], [9, 13], [10, 11], [10, 12], [10, 14], [11, 12], [11, 13], [11, 15], [12, 13], [12, 14], [12, 16], [13, 14], [13, 15], [14, 15], [14, 16], [15, 16]], 'homogeneous_cap': 4, 'minimum_vertices': 8}

def construct():
    n = P['n']
    edges = P['edges']
    
    # Build adjacency sets and bitmasks
    adj_sets = [set() for _ in range(n)]
    for u, v in edges:
        adj_sets[u].add(v)
        adj_sets[v].add(u)
    
    # Precompute degree of each vertex in the full graph
    full_deg = [len(adj_sets[i]) for i in range(n)]
    
    # For each pair, precompute common neighbors and non-common neighbors (within the full graph)
    # We'll use this for incremental updates
    
    best_score = -1
    best_vertices = None
    deadline = time.time() + 85.0
    
    # Generate all combinations of size 8 to 17
    # Start with larger sizes first since they can have more distinct degrees
    all_combos = []
    for k in range(n, 7, -1):  # 17 down to 8
        all_combos.extend(itertools.combinations(range(n), k))
    
    # Shuffle to get a more diverse search order, but keep larger sizes first
    import random
    random.seed(42)
    # Actually, let's just iterate in order but with a time check
    
    # Precompute for each vertex pair: which other vertices are adjacent to both, and to exactly one
    # For efficiency, let's use bitmasks
    adj_mask = [0] * n
    for u in range(n):
        for v in adj_sets[u]:
            adj_mask[u] |= (1 << v)
    
    # Iterate over combinations
    for k in range(n, 7, -1):
        for combo in itertools.combinations(range(n), k):
            if time.time() > deadline:
                break
            
            mask = 0
            for v in combo:
                mask |= (1 << v)
            
            # Compute degrees in induced subgraph
            degrees = set()
            for v in combo:
                deg = (adj_mask[v] & mask).bit_count()
                degrees.add(deg)
            
            score = len(degrees)
            if score > best_score:
                best_score = score
                best_vertices = list(combo)
                # If we have k distinct degrees in a k-vertex subgraph, that's perfect
                if score == k:
                    # Can't do better than k for a k-vertex subgraph
                    # But a larger subgraph might have more distinct degrees
                    # However, max distinct degrees is bounded by the range [0, k-1]
                    # So if score == k, we've achieved the max for this k
                    pass
        if time.time() > deadline:
            break
    
    # If no valid combination was found (shouldn't happen), fallback
    if best_vertices is None:
        best_vertices = list(range(8))
    
    # Ensure we return a list of Python ints
    return [int(v) for v in best_vertices]
# EVOLVE-BLOCK-END
