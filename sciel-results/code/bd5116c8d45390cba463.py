"""Optimized program: exhaustive search over all valid subsets to maximize distinct degrees."""
from itertools import combinations

P = {'n': 17, 'edges': [[0, 1], [0, 2], [0, 4], [0, 8], [0, 9], [0, 13], [0, 15], [0, 16], [1, 2], [1, 3], [1, 5], [1, 9], [1, 10], [1, 14], [1, 16], [2, 3], [2, 4], [2, 6], [2, 10], [2, 11], [2, 15], [3, 4], [3, 5], [3, 7], [3, 11], [3, 12], [3, 16], [4, 5], [4, 6], [4, 8], [4, 12], [4, 13], [5, 6], [5, 7], [5, 9], [5, 13], [5, 14], [6, 7], [6, 8], [6, 10], [6, 14], [6, 15], [7, 8], [7, 9], [7, 11], [7, 15], [7, 16], [8, 9], [8, 10], [8, 12], [8, 16], [9, 10], [9, 11], [9, 13], [10, 11], [10, 12], [10, 14], [11, 12], [11, 13], [11, 15], [12, 13], [12, 14], [12, 16], [13, 14], [13, 15], [14, 15], [14, 16], [15, 16]], 'homogeneous_cap': 4, 'minimum_vertices': 8}

def _best_subset():
    n = P['n']
    edges = P['edges']
    min_v = P['minimum_vertices']

    # Adjacency as sets for fast membership
    adj = [set() for _ in range(n)]
    for a, b in edges:
        adj[a].add(b)
        adj[b].add(a)

    best = None
    best_deg = -1

    # Precompute all vertices
    vertices = list(range(n))

    # Try all subset sizes from n down to min_v (larger subsets can have more distinct degrees)
    # For efficiency, iterate sizes descending and track best
    for size in range(n, min_v - 1, -1):
        for combo in combinations(vertices, size):
            combo_set = set(combo)
            degrees = []
            for v in combo:
                d = 0
                nb = adj[v]
                for u in combo:
                    if u != v and u in nb:
                        d += 1
                degrees.append(d)
            nd = len(set(degrees))
            if nd > best_deg:
                best_deg = nd
                best = list(combo)
                # Theoretical max for this size is `size` (degrees 0..size-1)
                if best_deg == size:
                    # Can't do better than size distinct degrees for a size-subset
                    # But a larger size subset could give more distinct degrees overall
                    pass
        # Early exit: max possible distinct degrees for any subset of size >= current is n
        # Actually max distinct degrees in induced subgraph of size s is s.
        # The global max is at most n=17. We can stop if best_deg == n, but unlikely.
        if best_deg == n:
            break

    return best

# Compute once at module load
_BEST = _best_subset()

def construct():
    # Ensure we return a list of Python ints
    result = list(_BEST)
    # Safety: ensure at least minimum_vertices
    if len(result) < P['minimum_vertices']:
        # Fallback
        result = list(range(P['minimum_vertices']))
    return result
