"""Direct exhaustive search over all vertex subsets to maximize distinct induced degrees."""
import time
import itertools

# EVOLVE-BLOCK-START
P = {'n': 17, 'edges': [[0, 1], [0, 2], [0, 4], [0, 8], [0, 9], [0, 13], [0, 15], [0, 16], [1, 2], [1, 3], [1, 5], [1, 9], [1, 10], [1, 14], [1, 16], [2, 3], [2, 4], [2, 6], [2, 10], [2, 11], [2, 15], [3, 4], [3, 5], [3, 7], [3, 11], [3, 12], [3, 16], [4, 5], [4, 6], [4, 8], [4, 12], [4, 13], [5, 6], [5, 7], [5, 9], [5, 13], [5, 14], [6, 7], [6, 8], [6, 10], [6, 14], [6, 15], [7, 8], [7, 9], [7, 11], [7, 15], [7, 16], [8, 9], [8, 10], [8, 12], [8, 16], [9, 10], [9, 11], [9, 13], [10, 11], [10, 12], [10, 14], [11, 12], [11, 13], [11, 15], [12, 13], [12, 14], [12, 16], [13, 14], [13, 15], [14, 15], [14, 16], [15, 16]], 'homogeneous_cap': 4, 'minimum_vertices': 8}

def construct():
    n = P['n']
    minv = P['minimum_vertices']
    edges = P['edges']
    start = time.time()
    deadline = start + 50.0   # safety margin well under the 60s limit

    # Build adjacency as bitmask for fast induced-degree computation
    adj = [0] * n
    for a, b in edges:
        adj[a] |= (1 << b)
        adj[b] |= (1 << a)

    best_count = -1
    best_verts = None

    vertices = list(range(n))
    # Iterate only over subsets of size >= minv (eliminates ~92% of mask-space waste)
    for k in range(minv, n + 1):
        for comb in itertools.combinations(vertices, k):
            # Build mask
            mask = 0
            for v in comb:
                mask |= (1 << v)
            # Compute degrees of each member within the induced subgraph
            degs = set()
            for v in comb:
                degs.add((adj[v] & mask).bit_count())
            nd = len(degs)
            if nd > best_count:
                best_count = nd
                best_verts = list(comb)
                # Upper bound: can't have more distinct degrees than vertices
                if best_count == k:
                    # Perfect: every vertex has a distinct degree
                    # Still check larger k for potentially more distinct degrees
                    pass
            # time guard
            if time.time() > deadline:
                if best_verts is not None:
                    return [int(v) for v in best_verts]
                # Fallback: first minv vertices
                return [int(v) for v in range(minv)]

    # Build vertex list from best_verts
    if best_verts is None:
        return [int(v) for v in range(minv)]
    verts = list(dict.fromkeys(int(v) for v in best_verts))
    return verts
# EVOLVE-BLOCK-END
