"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
from itertools import combinations

P = {'n': 17, 'edges': [[0, 1], [0, 2], [0, 4], [0, 8], [0, 9], [0, 13], [0, 15], [0, 16], [1, 2], [1, 3], [1, 5], [1, 9], [1, 10], [1, 14], [1, 16], [2, 3], [2, 4], [2, 6], [2, 10], [2, 11], [2, 15], [3, 4], [3, 5], [3, 7], [3, 11], [3, 12], [3, 16], [4, 5], [4, 6], [4, 8], [4, 12], [4, 13], [5, 6], [5, 7], [5, 9], [5, 13], [5, 14], [6, 7], [6, 8], [6, 10], [6, 14], [6, 15], [7, 8], [7, 9], [7, 11], [7, 15], [7, 16], [8, 9], [8, 10], [8, 12], [8, 16], [9, 10], [9, 11], [9, 13], [10, 11], [10, 12], [10, 14], [11, 12], [11, 13], [11, 15], [12, 13], [12, 14], [12, 16], [13, 14], [13, 15], [14, 15], [14, 16], [15, 16]], 'homogeneous_cap': 4, 'minimum_vertices': 8}

def construct():
    n = P['n']
    edges = P['edges']
    min_v = P['minimum_vertices']
    
    # Build adjacency set for quick lookup
    adj = set()
    for u, v in edges:
        adj.add((u, v))
        adj.add((v, u))
    
    best_count = 0
    best_subset = None
    
    # Try all subset sizes from min_v to n
    for size in range(min_v, n + 1):
        for subset in combinations(range(n), size):
            subset_set = set(subset)
            # Compute degree of each vertex in the induced subgraph
            degrees = []
            for v in subset:
                deg = 0
                for u in subset:
                    if u != v and (v, u) in adj:
                        deg += 1
                degrees.append(deg)
            distinct = len(set(degrees))
            if distinct > best_count:
                best_count = distinct
                best_subset = subset
                # Early termination: can't do better than size distinct degrees
                if best_count == size:
                    break
        # If we achieved max possible for this size, check if larger size can beat it
        if best_count == size:
            # A larger subset could have more distinct degrees, so continue
            pass
        else:
            # Even the best for this size is less than size, but larger sizes might still improve
            # We can't easily prune, so continue
            pass
    
    # Ensure we return a list of Python ints
    result = [int(v) for v in best_subset]
    return result
# EVOLVE-BLOCK-END
