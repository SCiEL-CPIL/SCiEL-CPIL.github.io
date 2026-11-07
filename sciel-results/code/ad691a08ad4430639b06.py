"""Optimized exhaustive bitmask search for maximum distinct degrees in induced subgraph."""
import time

P = {'n': 17, 'edges': [[0, 1], [0, 2], [0, 4], [0, 8], [0, 9], [0, 13], [0, 15], [0, 16], [1, 2], [1, 3], [1, 5], [1, 9], [1, 10], [1, 14], [1, 16], [2, 3], [2, 4], [2, 6], [2, 10], [2, 11], [2, 15], [3, 4], [3, 5], [3, 7], [3, 11], [3, 12], [3, 16], [4, 5], [4, 6], [4, 8], [4, 12], [4, 13], [5, 6], [5, 7], [5, 9], [5, 13], [5, 14], [6, 7], [6, 8], [6, 10], [6, 14], [6, 15], [7, 8], [7, 9], [7, 11], [7, 15], [7, 16], [8, 9], [8, 10], [8, 12], [8, 16], [9, 10], [9, 11], [9, 13], [10, 11], [10, 12], [10, 14], [11, 12], [11, 13], [11, 15], [12, 13], [12, 14], [12, 16], [13, 14], [13, 15], [14, 15], [14, 16], [15, 16]], 'homogeneous_cap': 4, 'minimum_vertices': 8}

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin out of 90s

    n = P['n']
    edges = P['edges']
    min_k = P['minimum_vertices']

    # Build adjacency bitmask for each vertex
    adj_mask = [0] * n
    for u, v in edges:
        adj_mask[u] |= (1 << v)
        adj_mask[v] |= (1 << u)

    best_subset = None
    max_distinct = 0

    # Iterate over all 2^n bitmasks; filter by popcount >= min_k
    # For each valid mask, compute distinct degrees in O(n) time
    for mask in range(1 << n):
        if time.time() > deadline:
            break

        k = mask.bit_count()
        if k < min_k:
            continue

        # Upper bound: can't have more than k distinct degrees
        if k <= max_distinct:
            continue

        # Compute degrees of each vertex in the induced subgraph
        degrees = set()
        for v in range(n):
            if mask & (1 << v):
                deg = (adj_mask[v] & mask).bit_count()
                degrees.add(deg)

        distinct_count = len(degrees)
        if distinct_count > max_distinct:
            max_distinct = distinct_count
            # Extract vertex list from mask
            best_subset = [v for v in range(n) if mask & (1 << v)]

            # Early exit: perfect score for this subset size
            if distinct_count == k:
                return [int(v) for v in best_subset]

    # Fallback
    if best_subset is None:
        best_subset = [int(v) for v in range(min_k)]

    return [int(v) for v in best_subset]
