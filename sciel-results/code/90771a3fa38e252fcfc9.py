"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from itertools import combinations

P = {'n': 17, 'edges': [[0, 1], [0, 2], [0, 4], [0, 8], [0, 9], [0, 13], [0, 15], [0, 16], [1, 2], [1, 3], [1, 5], [1, 9], [1, 10], [1, 14], [1, 16], [2, 3], [2, 4], [2, 6], [2, 10], [2, 11], [2, 15], [3, 4], [3, 5], [3, 7], [3, 11], [3, 12], [3, 16], [4, 5], [4, 6], [4, 8], [4, 12], [4, 13], [5, 6], [5, 7], [5, 9], [5, 13], [5, 14], [6, 7], [6, 8], [6, 10], [6, 14], [6, 15], [7, 8], [7, 9], [7, 11], [7, 15], [7, 16], [8, 9], [8, 10], [8, 12], [8, 16], [9, 10], [9, 11], [9, 13], [10, 11], [10, 12], [10, 14], [11, 12], [11, 13], [11, 15], [12, 13], [12, 14], [12, 16], [13, 14], [13, 15], [14, 15], [14, 16], [15, 16]], 'homogeneous_cap': 4, 'minimum_vertices': 8}

def construct():
    n = P['n']
    edges = P['edges']
    min_v = P['minimum_vertices']
    
    # Build adjacency as bitmask for fast computation
    adj = [0] * n
    for u, v in edges:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
    
    deadline = time.time() + 80  # safety margin before 90s kill
    
    best_subset = list(range(min_v))
    best_score = -1
    
    # Precompute all subsets of size >= min_v
    # For each subset, compute distinct degrees in induced subgraph
    for size in range(min_v, n + 1):
        if time.time() > deadline:
            break
        for subset in combinations(range(n), size):
            if time.time() > deadline:
                break
            mask = 0
            for v in subset:
                mask |= (1 << v)
            
            # Compute degrees in induced subgraph
            degrees = set()
            for v in subset:
                # Count neighbors of v that are in subset
                deg = bin(adj[v] & mask).count('1')
                degrees.add(deg)
            
            score = len(degrees)
            if score > best_score:
                best_score = score
                best_subset = list(subset)
    
    # Ensure we return Python ints in a list
    return [int(v) for v in best_subset]
# EVOLVE-BLOCK-END
