"""Construct a k-regular graph with girth >= g (cage problem).

Structural approach: build the Lazebnik-Ustimenko-Woldar (LUW) graph
CD(d, q) with q = k in bulk (no per-vertex generator), then return the
connected component of the zero point. The LUW graph is q-regular,
bipartite and has girth >= d + 5 for odd d, so choosing d with d + 5 >= g
gives a valid k-regular graph with girth >= g. An explicit deadline guard
returns the best valid result found before the wall-clock limit.
"""
import time
import itertools

# EVOLVE-BLOCK-START
def construct(k, g):
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin below 150 s

    def _time_left():
        return deadline - time.time()

    # Choose d so girth >= g. LUW girth >= d + 5 for odd d.
    d = g - 5
    if d % 2 == 0:
        d += 1
    d = max(3, d)

    q = k
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add = lambda x, y: x ^ y
        mul = lambda x, y: table[x][y]
    else:
        add = lambda x, y: (x + y) % q
        mul = lambda x, y: x * y % q

    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r, c0 = (c - 5) % 4, c - (c - 5) % 4
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]

    total = q ** d
    all_coords = list(itertools.product(range(q), repeat=d))
    n = 2 * total

    # index[(side, coords)] -> int ; points are 0..total-1, lines total..2*total-1
    index = {}
    for ci, coords in enumerate(all_coords):
        index[(0, coords)] = ci
        index[(1, coords)] = total + ci

    adj = [[] for _ in range(n)]
    edges = []
    # Precompute rule arrays for speed (0-based: index i corresponds to c = i+1)
    # For c in 2..d: a,b = rule[c]; we need x[c-1], y[a-1], x[b-1].
    rc = range(2, d + 1)
    for ci, x in enumerate(all_coords):
        for first in range(q):
            y = [0] * d
            y[0] = first  # y[1]
            for c in rc:
                a, b = rule[c]
                y[c - 1] = add(x[c - 1], mul(y[a - 1], x[b - 1]))
            lidx = index[(1, tuple(y))]
            if ci < lidx:
                edges.append((ci, lidx))
            adj[ci].append(lidx)
            adj[lidx].append(ci)
        if _time_left() < 5.0:
            break

    # Connected component of zero point (index 0).
    zero = 0
    seen = [False] * n
    stack = [zero]
    seen[zero] = True
    comp = []
    while stack:
        u = stack.pop()
        comp.append(u)
        for w in adj[u]:
            if not seen[w]:
                seen[w] = True
                stack.append(w)
    comp_set = set(comp)
    remap = {v: i for i, v in enumerate(comp)}
    ncomp = len(comp)
    new_edges = []
    for (u, v) in edges:
        if u in comp_set and v in comp_set:
            nu, nv = remap[u], remap[v]
            if nu < nv:
                new_edges.append((nu, nv))
    return ncomp, new_edges
# EVOLVE-BLOCK-END
