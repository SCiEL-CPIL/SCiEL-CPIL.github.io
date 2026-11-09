"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import time

P = {'n': 17, 'edges': [[0, 1], [0, 2], [0, 4], [0, 8], [0, 9], [0, 13], [0, 15], [0, 16], [1, 2], [1, 3], [1, 5], [1, 9], [1, 10], [1, 14], [1, 16], [2, 3], [2, 4], [2, 6], [2, 10], [2, 11], [2, 15], [3, 4], [3, 5], [3, 7], [3, 11], [3, 12], [3, 16], [4, 5], [4, 6], [4, 8], [4, 12], [4, 13], [5, 6], [5, 7], [5, 9], [5, 13], [5, 14], [6, 7], [6, 8], [6, 10], [6, 14], [6, 15], [7, 8], [7, 9], [7, 11], [7, 15], [7, 16], [8, 9], [8, 10], [8, 12], [8, 16], [9, 10], [9, 11], [9, 13], [10, 11], [10, 12], [10, 14], [11, 12], [11, 13], [11, 15], [12, 13], [12, 14], [12, 16], [13, 14], [13, 15], [14, 15], [14, 16], [15, 16]], 'homogeneous_cap': 4, 'minimum_vertices': 8}

def construct():
    n = P['n']
    edges = P['edges']
    min_verts = P['minimum_vertices']
    
    # Build adjacency set for fast lookup
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    
    deadline = time.time() + 80  # safety margin before 90s limit
    best_subset = None
    best_distinct = 0
    
    # Try all subset sizes from largest to smallest (larger subsets can have more distinct degrees)
    for size in range(n, min_verts - 1, -1):
        if time.time() > deadline:
            break
        for combo in itertools.combinations(range(n), size):
            if time.time() > deadline:
                break
            combo_set = set(combo)
            # Compute degrees in induced subgraph
            degrees = []
            for v in combo:
                deg = sum(1 for u in combo_set if u in adj[v])
                degrees.append(deg)
            distinct_count = len(set(degrees))
            if distinct_count > best_distinct:
                best_distinct = distinct_count
                best_subset = list(combo)
                # Early termination: can't do better than size distinct degrees
                if best_distinct == size:
                    # Check if we can get even more with larger size
                    # Actually size is decreasing, so we found max for this size
                    # But larger sizes were already tried, so this is optimal
                    break
        if best_distinct == size:
            # We found a perfect spread, but larger sizes were already checked
            # Actually we're going from large to small, so if we found perfect at size,
            # larger sizes had worse or equal. But we want max distinct, so continue
            # unless we hit the theoretical max for any size
            pass
    
    if best_subset is None:
        # Fallback
        best_subset = list(range(min_verts))
    
    return list(best_subset)
# EVOLVE-BLOCK-END
