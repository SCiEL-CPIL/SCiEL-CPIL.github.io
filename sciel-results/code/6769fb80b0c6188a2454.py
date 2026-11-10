"""Few colors without any bicolored cycle: contiguous-block coloring of the 50-cycle."""
# EVOLVE-BLOCK-START
import time

P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}

DEADLINE = 80.0  # safety margin well under the 90 s kill limit


def _has_bicolored_cycle(coloring):
    """Union-find check: for every pair of colors, the induced subgraph is a forest.
    Returns True if ANY bicolored cycle exists (i.e. coloring is INVALID)."""
    n = P['n']
    edges = P['edges']
    # group vertices by color
    by_color = {}
    for v in range(n):
        c = coloring[v]
        by_color.setdefault(c, []).append(v)
    colors = sorted(by_color.keys())
    # For each color pair, build induced edges and detect a cycle via union-find.
    # An induced subgraph on a vertex set has a cycle iff an edge connects two vertices
    # already in the same component while processing that subgraph's edges.
    # To be efficient, we only need to consider edges whose both endpoints share a color pair.
    # For each edge (u,v) with cu != cv, it belongs to the subgraph of pair {cu,cv}.
    # We collect edges per pair and run union-find per pair.
    pair_edges = {}
    for (u, v) in edges:
        cu = coloring[u]
        cv = coloring[v]
        if cu == cv:
            # same color edge is fine (not bicolored), but it means adjacent same color -> invalid adjacency.
            # Actually adjacent vertices MUST differ, so this is invalid.
            return True
        key = (cu, cv) if cu < cv else (cv, cu)
        pair_edges.setdefault(key, []).append((u, v))
    for key, elist in pair_edges.items():
        parent = {}
        rank = {}

        def find(x):
            if x not in parent:
                parent[x] = x
                rank[x] = 0
            r = x
            while parent[r] != r:
                r = parent[r]
            # path compression
            while parent[x] != r:
                nxt = parent[x]
                parent[x] = r
                x = nxt
            return r

        for (u, v) in elist:
            ru = find(u)
            rv = find(v)
            if ru == rv:
                return True  # cycle in this bicolored subgraph
            if rank[ru] < rank[rv]:
                ru, rv = rv, ru
            parent[rv] = ru
            if rank[ru] == rank[rv]:
                rank[ru] += 1
    return False


def _contiguous_block_coloring(n, k):
    """Assign k colors to the n cycle positions as k contiguous blocks.
    Vertex i gets color floor(i * k / n). This makes each color class a contiguous
    arc of the cycle, so any two colors induce only disjoint paths (a forest)."""
    coloring = [0] * n
    for i in range(n):
        coloring[i] = min(k - 1, (i * k) // n)
    return coloring


def construct():
    start = time.time()
    n = P['n']
    # Start with a guaranteed-valid 2-color contiguous-block coloring.
    best = _contiguous_block_coloring(n, 2)
    if not _has_bicolored_cycle(best):
        return [int(c) for c in best]

    # Fallback (should not trigger): try more colors.
    for k in range(3, n + 1):
        if time.time() - start > DEADLINE:
            break
        cand = _contiguous_block_coloring(n, k)
        if not _has_bicolored_cycle(cand):
            best = cand
            break
    return [int(c) for c in best]
# EVOLVE-BLOCK-END
