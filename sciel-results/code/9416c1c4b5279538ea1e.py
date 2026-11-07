"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from itertools import combinations

P = {'n': 17, 'edges': [[0, 1], [0, 2], [0, 4], [0, 8], [0, 9], [0, 13], [0, 15], [0, 16], [1, 2], [1, 3], [1, 5], [1, 9], [1, 10], [1, 14], [1, 16], [2, 3], [2, 4], [2, 6], [2, 10], [2, 11], [2, 15], [3, 4], [3, 5], [3, 7], [3, 11], [3, 12], [3, 16], [4, 5], [4, 6], [4, 8], [4, 12], [4, 13], [5, 6], [5, 7], [5, 9], [5, 13], [5, 14], [6, 7], [6, 8], [6, 10], [6, 14], [6, 15], [7, 8], [7, 9], [7, 11], [7, 15], [7, 16], [8, 9], [8, 10], [8, 12], [8, 16], [9, 10], [9, 11], [9, 13], [10, 11], [10, 12], [10, 14], [11, 12], [11, 13], [11, 15], [12, 13], [12, 14], [12, 16], [13, 14], [13, 15], [14, 15], [14, 16], [15, 16]], 'homogeneous_cap': 4, 'minimum_vertices': 8}

# Precompute adjacency as a set of frozensets for fast lookup
_edges = P['edges']
_adj = [set() for _ in range(17)]
for a, b in _edges:
    _adj[a].add(b)
    _adj[b].add(a)

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

def construct():
    deadline = time.time() + 70  # safety margin before 90s
    best_score = -1
    best_subset = None
    
    n = 17
    # We need at least 8 vertices. To get many distinct degrees, larger subsets help.
    # Enumerate all subsets of size 8 to 17.
    for k in range(8, n + 1):
        for subset in combinations(range(n), k):
            if time.time() > deadline:
                break
            score = _distinct_degrees(subset)
            if score > best_score:
                best_score = score
                best_subset = subset
            # Early termination: max possible for this k is k
            if best_score >= n:  # can't do better than 17 distinct degrees, but max is k
                break
        if time.time() > deadline:
            break
        # If we found k distinct degrees for some k, and larger k could give more,
        # but if best_score == k for current k, larger k might give k+1, etc.
        # Keep going to find the global max.
    
    if best_subset is None:
        return list(range(8))
    
    # Return as list of Python ints
    return [int(v) for v in best_subset]
# EVOLVE-BLOCK-END
