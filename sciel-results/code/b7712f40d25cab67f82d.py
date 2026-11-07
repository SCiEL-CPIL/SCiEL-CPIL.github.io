"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

P = {'n': 17, 'edges': [[0, 1], [0, 2], [0, 4], [0, 8], [0, 9], [0, 13], [0, 15], [0, 16], [1, 2], [1, 3], [1, 5], [1, 9], [1, 10], [1, 14], [1, 16], [2, 3], [2, 4], [2, 6], [2, 10], [2, 11], [2, 15], [3, 4], [3, 5], [3, 7], [3, 11], [3, 12], [3, 16], [4, 5], [4, 6], [4, 8], [4, 12], [4, 13], [5, 6], [5, 7], [5, 9], [5, 13], [5, 14], [6, 7], [6, 8], [6, 10], [6, 14], [6, 15], [7, 8], [7, 9], [7, 11], [7, 15], [7, 16], [8, 9], [8, 10], [8, 12], [8, 16], [9, 10], [9, 11], [9, 13], [10, 11], [10, 12], [10, 14], [11, 12], [11, 13], [11, 15], [12, 13], [12, 14], [12, 16], [13, 14], [13, 15], [14, 15], [14, 16], [15, 16]], 'homogeneous_cap': 4, 'minimum_vertices': 8}

def _best_vertices():
    n = int(P['n'])
    minv = int(P.get('minimum_vertices', 8))
    # adjacency bitmask per vertex
    adj = [0] * n
    for a, b in P['edges']:
        adj[a] |= (1 << b)
        adj[b] |= (1 << a)
    popcnt = [0] * (1 << n)
    best = None
    best_score = -1
    # enumerate all masks; incrementally update degree counts of the induced subgraph
    for mask in range(1 << n):
        pc = popcnt[mask >> 1] + (mask & 1)
        popcnt[mask] = pc
        if pc < minv:
            continue
        # compute induced degrees: for each selected vertex, count selected neighbors
        degs = {}
        m = mask
        while m:
            lsb = m & (-m)
            i = lsb.bit_length() - 1
            d = (adj[i] & mask).bit_count()
            degs[d] = degs.get(d, 0) + 1
            m ^= lsb
        s = len(degs)
        if s > best_score:
            best_score = s
            best = [i for i in range(n) if (mask >> i) & 1]
    return best

def construct():
    start = time.time()
    deadline = start + 80.0
    result = _best_vertices()
    if result is None:
        result = list(range(int(P.get('minimum_vertices', 8))))
    # enforce types: Python ints in a list
    out = [int(v) for v in result]
    if time.time() > deadline:
        return out
    return out
# EVOLVE-BLOCK-END
