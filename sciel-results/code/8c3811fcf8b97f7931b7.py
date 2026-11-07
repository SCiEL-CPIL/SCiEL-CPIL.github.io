"""Optimized complete search with precomputed degrees."""
# EVOLVE-BLOCK-START
import time

P = {'n': 17, 'edges': [[0, 1], [0, 2], [0, 4], [0, 8], [0, 9], [0, 13], [0, 15], [0, 16], [1, 2], [1, 3], [1, 5], [1, 9], [1, 10], [1, 14], [1, 16], [2, 3], [2, 4], [2, 6], [2, 10], [2, 11], [2, 15], [3, 4], [3, 5], [3, 7], [3, 11], [3, 12], [3, 16], [4, 5], [4, 6], [4, 8], [4, 12], [4, 13], [5, 6], [5, 7], [5, 9], [5, 13], [5, 14], [6, 7], [6, 8], [6, 10], [6, 14], [6, 15], [7, 8], [7, 9], [7, 11], [7, 15], [7, 16], [8, 9], [8, 10], [8, 12], [8, 16], [9, 10], [9, 11], [9, 13], [10, 11], [10, 12], [10, 14], [11, 12], [11, 13], [11, 15], [12, 13], [12, 14], [12, 16], [13, 14], [13, 15], [14, 15], [14, 16], [15, 16]], 'homogeneous_cap': 4, 'minimum_vertices': 8}


def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin
    
    n = P['n']
    edges = P['edges']
    minv = P['minimum_vertices']

    # Precompute adjacency bitmasks
    adj = [0] * n
    for a, b in edges:
        adj[a] |= (1 << b)
        adj[b] |= (1 << a)

    # Precompute degrees for each vertex in each mask
    # deg[v][mask] = number of neighbors of v in mask
    # This is 17 * 2^17 = 2,228,224 entries.
    # Storing as a list of lists of small ints is memory efficient enough.
    deg = [[0] * (1 << n) for _ in range(n)]
    
    for v in range(n):
        dv = deg[v]
        av = adj[v]
        # Iterate through all masks
        # Optimization: dv[mask] = dv[mask_without_lsb] + (1 if lsb is neighbor else 0)
        # This is O(2^n) per vertex.
        for mask in range(1, 1 << n):
            # Get lowest set bit
            lsb = mask & -mask
            # Previous mask without lsb
            prev_mask = mask ^ lsb
            # Vertex index of lsb
            u = lsb.bit_length() - 1
            # If u is a neighbor of v, degree increases by 1
            # Check if bit u is set in adj[v]
            if av & (1 << u):
                dv[mask] = dv[prev_mask] + 1
            else:
                dv[mask] = dv[prev_mask]

    best_count = -1
    best_mask = 0

    # Iterate through all masks
    for mask in range(1 << n):
        if time.time() > deadline:
            break
            
        # Check size
        if mask.bit_count() < minv:
            continue

        # Get distinct degrees
        # We only care about vertices in the mask
        degs = set()
        m = mask
        while m:
            lsb = m & -m
            v = lsb.bit_length() - 1
            degs.add(deg[v][mask])
            m ^= lsb
            
        c = len(degs)
        if c > best_count:
            best_count = c
            best_mask = mask
            # If we found a subset of size k with k distinct degrees, 
            # we can't do better than k for this size.
            # But we might find a larger subset with more distinct degrees.
            # Max possible is n.
            if best_count == n:
                break

    # Extract vertices from best_mask
    best_vertices = []
    m = best_mask
    while m:
        lsb = m & -m
        v = lsb.bit_length() - 1
        best_vertices.append(int(v))
        m ^= lsb

    return best_vertices
# EVOLVE-BLOCK-END
