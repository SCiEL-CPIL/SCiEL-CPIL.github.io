"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

P = {'n': 17, 'edges': [[0, 1], [0, 2], [0, 4], [0, 8], [0, 9], [0, 13], [0, 15], [0, 16], [1, 2], [1, 3], [1, 5], [1, 9], [1, 10], [1, 14], [1, 16], [2, 3], [2, 4], [2, 6], [2, 10], [2, 11], [2, 15], [3, 4], [3, 5], [3, 7], [3, 11], [3, 12], [3, 16], [4, 5], [4, 6], [4, 8], [4, 12], [4, 13], [5, 6], [5, 7], [5, 9], [5, 13], [5, 14], [6, 7], [6, 8], [6, 10], [6, 14], [6, 15], [7, 8], [7, 9], [7, 11], [7, 15], [7, 16], [8, 9], [8, 10], [8, 12], [8, 16], [9, 10], [9, 11], [9, 13], [10, 11], [10, 12], [10, 14], [11, 12], [11, 13], [11, 15], [12, 13], [12, 14], [12, 16], [13, 14], [13, 15], [14, 15], [14, 16], [15, 16]], 'homogeneous_cap': 4, 'minimum_vertices': 8}

def construct():
    n = P['n']
    min_v = P['minimum_vertices']
    deadline = time.time() + 85.0

    # Build adjacency bitmasks
    adj = [0] * n
    for u, v in P['edges']:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)

    best_value = -1
    best_combo = None
    all_vertices = list(range(n))

    # Try sizes from largest to smallest; skip sizes that cannot beat current best
    for size in range(n, min_v - 1, -1):
        if best_value >= size:
            # Cannot improve with this size or any smaller size
            break
        
        # Start with the smallest mask of 'size' bits
        mask = (1 << size) - 1
        limit = 1 << n
        while mask < limit:
            if time.time() > deadline:
                return [int(v) for v in all_vertices if (best_combo >> v) & 1] if best_combo is not None else [int(v) for v in all_vertices[:min_v]]
            
            # Compute distinct degrees for vertices in mask
            deg_set = set()
            m = mask
            while m:
                lsb = m & (-m)
                v = lsb.bit_length() - 1
                deg = (adj[v] & mask).bit_count()
                deg_set.add(deg)
                m ^= lsb
            
            val = len(deg_set)
            if val > best_value:
                best_value = val
                best_combo = mask
                if best_value == size:
                    # Max possible for this size; break to try smaller sizes
                    break
            
            # Next combination via Gosper's hack
            c = mask & (-mask)
            r = mask + c
            if r >= limit:
                break
            mask = (((r ^ mask) >> 2) // c) | r

    if best_combo is None:
        return [int(v) for v in all_vertices[:min_v]]

    result = [int(v) for v in all_vertices if (best_combo >> v) & 1]
    return result
# EVOLVE-BLOCK-END
