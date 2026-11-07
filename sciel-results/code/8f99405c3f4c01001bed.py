"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from itertools import combinations

P = {'n': 17, 'edges': [[0, 1], [0, 2], [0, 4], [0, 8], [0, 9], [0, 13], [0, 15], [0, 16], [1, 2], [1, 3], [1, 5], [1, 9], [1, 10], [1, 14], [1, 16], [2, 3], [2, 4], [2, 6], [2, 10], [2, 11], [2, 15], [3, 4], [3, 5], [3, 7], [3, 11], [3, 12], [3, 16], [4, 5], [4, 6], [4, 8], [4, 12], [4, 13], [5, 6], [5, 7], [5, 9], [5, 13], [5, 14], [6, 7], [6, 8], [6, 10], [6, 14], [6, 15], [7, 8], [7, 9], [7, 11], [7, 15], [7, 16], [8, 9], [8, 10], [8, 12], [8, 16], [9, 10], [9, 11], [9, 13], [10, 11], [10, 12], [10, 14], [11, 12], [11, 13], [11, 15], [12, 13], [12, 14], [12, 16], [13, 14], [13, 15], [14, 15], [14, 16], [15, 16]], 'homogeneous_cap': 4, 'minimum_vertices': 8}

# Precompute adjacency as a set of frozensets for fast lookup
_edges = P['edges']
_n = P['n']
_adj = [set() for _ in range(_n)]
for a, b in _edges:
    _adj[a].add(b)
    _adj[b].add(a)

# Precompute adjacency bitmask for each vertex
_adj_mask = [0] * _n
for a, b in _edges:
    _adj_mask[a] |= (1 << b)
    _adj_mask[b] |= (1 << a)

def _distinct_degrees(subset):
    """Compute number of distinct degrees in induced subgraph on subset."""
    s = set(subset)
    degs = []
    for v in subset:
        d = 0
        for u in subset:
            if u != v and u in _adj[v]:
                d += 1
        degs.append(d)
    return len(set(degs))

def _count_distinct_degrees_mask(mask):
    """Compute number of distinct degrees using bitmask representation."""
    if mask == 0:
        return 0
    # Get list of vertices in mask
    vertices = [v for v in range(_n) if (mask >> v) & 1]
    k = len(vertices)
    degs = []
    for v in vertices:
        # Count neighbors of v within the mask
        d = bin(_adj_mask[v] & mask).count('1')
        degs.append(d)
    return len(set(degs))

def construct():
    deadline = time.time() + 75  # safety margin before 90s
    best_score = -1
    best_subset = None
    
    # Exhaustive search over all subsets of size >= 8
    # Using bitmask for efficiency
    for k in range(8, _n + 1):
        for subset in combinations(range(_n), k):
            if time.time() > deadline:
                break
            # Convert to mask
            mask = 0
            for v in subset:
                mask |= (1 << v)
            score = _count_distinct_degrees_mask(mask)
            if score > best_score:
                best_score = score
                best_subset = subset
            # Early termination: max possible for this k is k
            if best_score >= _n:
                break
        if time.time() > deadline:
            break
    
    if best_subset is None:
        return list(range(8))
    
    # Return as list of Python ints
    return [int(v) for v in best_subset]
# EVOLVE-BLOCK-END
