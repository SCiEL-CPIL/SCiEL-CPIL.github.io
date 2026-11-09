"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import time

P = {'n': 17, 'edges': [[0, 1], [0, 2], [0, 4], [0, 8], [0, 9], [0, 13], [0, 15], [0, 16], [1, 2], [1, 3], [1, 5], [1, 9], [1, 10], [1, 14], [1, 16], [2, 3], [2, 4], [2, 6], [2, 10], [2, 11], [2, 15], [3, 4], [3, 5], [3, 7], [3, 11], [3, 12], [3, 16], [4, 5], [4, 6], [4, 8], [4, 12], [4, 13], [5, 6], [5, 7], [5, 9], [5, 13], [5, 14], [6, 7], [6, 8], [6, 10], [6, 14], [6, 15], [7, 8], [7, 9], [7, 11], [7, 15], [7, 16], [8, 9], [8, 10], [8, 12], [8, 16], [9, 10], [9, 11], [9, 13], [10, 11], [10, 12], [10, 14], [11, 12], [11, 13], [11, 15], [12, 13], [12, 14], [12, 16], [13, 14], [13, 15], [14, 15], [14, 16], [15, 16]], 'homogeneous_cap': 4, 'minimum_vertices': 8}


def construct():
    n = P['n']
    edges = P['edges']
    min_vertices = P['minimum_vertices']

    # Build adjacency set for fast lookup
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)

    deadline = time.time() + 80  # safety margin before 90s kill

    best_value = -1
    best_subset = None

    # Enumerate all subsets of size >= min_vertices
    for size in range(min_vertices, n + 1):
        if time.time() > deadline:
            break
        for subset in itertools.combinations(range(n), size):
            if time.time() > deadline:
                break
            subset_set = set(subset)
            # Compute degrees within induced subgraph
            degrees = []
            for v in subset:
                deg = sum(1 for u in subset_set if u != v and u in adj[v])
                degrees.append(deg)
            distinct = len(set(degrees))
            if distinct > best_value:
                best_value = distinct
                best_subset = list(subset)
            # Early termination: can't do better than size distinct degrees
            if best_value == size:
                # Perfect: all degrees distinct in this size subset
                # Check if larger sizes could do better
                if best_value >= n - 1:
                    return best_subset

    if best_subset is None:
        return list(range(min_vertices))

    return best_subset
# EVOLVE-BLOCK-END
