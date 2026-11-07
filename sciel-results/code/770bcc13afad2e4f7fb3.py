"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import itertools

P = {'n': 17, 'edges': [[0, 1], [0, 2], [0, 4], [0, 8], [0, 9], [0, 13], [0, 15], [0, 16], [1, 2], [1, 3], [1, 5], [1, 9], [1, 10], [1, 14], [1, 16], [2, 3], [2, 4], [2, 6], [2, 10], [2, 11], [2, 15], [3, 4], [3, 5], [3, 7], [3, 11], [3, 12], [3, 16], [4, 5], [4, 6], [4, 8], [4, 12], [4, 13], [5, 6], [5, 7], [5, 9], [5, 13], [5, 14], [6, 7], [6, 8], [6, 10], [6, 14], [6, 15], [7, 8], [7, 9], [7, 11], [7, 15], [7, 16], [8, 9], [8, 10], [8, 12], [8, 16], [9, 10], [9, 11], [9, 13], [10, 11], [10, 12], [10, 14], [11, 12], [11, 13], [11, 15], [12, 13], [12, 14], [12, 16], [13, 14], [13, 15], [14, 15], [14, 16], [15, 16]], 'homogeneous_cap': 4, 'minimum_vertices': 8}

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin for 90s limit

    n = P['n']
    edges = P['edges']
    min_vertices = P['minimum_vertices']

    # Precompute adjacency bitmasks for fast degree calculation within subsets
    adj = [0] * n
    for u, v in edges:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)

    # Determine the theoretical maximum number of distinct degrees
    # The degree of any vertex in an induced subgraph is at most its degree in the full graph.
    max_full_deg = 0
    for u in range(n):
        deg = adj[u].bit_count()
        if deg > max_full_deg:
            max_full_deg = deg

    # Max distinct degrees is at most max_full_deg + 1 (values 0 to max_full_deg)
    target_max = max_full_deg + 1

    best_score = -1
    best_subset = None

    # Enumerate all subsets of size at least min_vertices
    # We iterate from largest size down to smallest, as larger subsets have more
    # potential for distinct degrees (up to 'size' distinct values).
    for size in range(n, min_vertices - 1, -1):
        if time.time() > deadline:
            break

        # If we already found the theoretical maximum, we can stop completely
        if best_score == target_max:
            break

        # Upper bound for this size: at most 'size' distinct degrees
        # If best_score >= size, no subset of this size can beat it
        if best_score >= size:
            continue

        for subset in itertools.combinations(range(n), size):
            if time.time() > deadline:
                break

            # Calculate mask for this subset
            mask = 0
            for v in subset:
                mask |= (1 << v)

            # Calculate distinct degrees
            deg_set = set()
            for v in subset:
                deg_set.add((adj[v] & mask).bit_count())

            distinct_degrees = len(deg_set)

            if distinct_degrees > best_score:
                best_score = distinct_degrees
                best_subset = list(subset)

                # Early exit if we hit the theoretical maximum possible distinct degrees
                if best_score == target_max:
                    return best_subset

    if best_subset is None:
        # Fallback to a valid subset if something went wrong (should not happen)
        return list(range(min_vertices))

    return best_subset
# EVOLVE-BLOCK-END
