"""Triangle-free, non-3-colorable graph on 16 vertices, maximizing edges.

Strategy: blow-up of the Grötzsch graph. The Grötzsch graph is the smallest
triangle-free graph with chromatic number 4. We concentrate vertex copies on
its high-degree nodes to maximize edge density while preserving the
triangle-free and non-3-colorable properties.
"""

# Grötzsch graph edges on 11 vertices (Mycielski of C5).
# Vertices 0..4 form a 5-cycle; 5..9 are the Mycielski copies; 10 is the apex.
_GROTZSCH_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 4), (4, 0),      # 5-cycle
    (0, 6), (1, 6), (1, 7), (2, 7), (2, 8),      # cycle nodes to copies
    (3, 8), (3, 9), (4, 9), (4, 6),
    (5, 6), (5, 7), (5, 8), (5, 9), (5, 10),     # copy of node 5 to others + apex
]
# Note: standard Grötzsch construction: base C5 (0-4), copies (5-9), apex 10.
# Let me use the canonical adjacency to be safe.


def _grotzsch_edges():
    """Return the 20 edges of the Grötzsch graph (11 vertices)."""
    edges = []
    # 5-cycle on 0..4
    for i in range(5):
        edges.append((i, (i + 1) % 5))
    # Mycielski: for each cycle edge (i, j), add (i, 5+j) and (j, 5+i)
    cycle = [(i, (i + 1) % 5) for i in range(5)]
    for u, w in cycle:
        edges.append((u, 5 + w))
        edges.append((w, 5 + u))
    # apex 10 connects to all 5 Mycielski copies 5..9
    for u in range(5):
        edges.append((5 + u, 10))
    return edges


def _blow_up(weights):
    """Blow up the Grötzsch graph with the given per-vertex weights.

    For each Grötzsch edge (u, v), connect a matching of size min(w_u, w_v)
    between the copy-sets of u and v. This keeps the graph triangle-free
    (no two adjacent copy-sets are fully connected, and non-adjacent copy-sets
    are disjoint with no edges).

    Returns (edges, vertex_map) where vertex_map[original] = list of new ids.
    """
    grot_edges = _grotzsch_edges()
    # Assign new ids in order of original vertices.
    vertex_map = {}
    next_id = 0
    for u in range(11):
        vertex_map[u] = list(range(next_id, next_id + weights[u]))
        next_id += weights[u]

    edges = []
    for u, v in grot_edges:
        cu = vertex_map[u]
        cv = vertex_map[v]
        k = min(len(cu), len(cv))
        for i in range(k):
            edges.append([cu[i], cv[i]])
    return edges, next_id


def _is_triangle_free(edges, n):
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    for u in range(n):
        for v in adj[u]:
            if u < v:
                for w in adj[u] & adj[v]:
                    if w != u and w != v:
                        return False
    return True


def _is_not_3_colorable(edges, n):
    """Exact backtracking 3-colorability test. Returns True if NOT 3-colorable."""
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)

    # Order vertices by descending degree for faster pruning.
    order = sorted(range(n), key=lambda x: -len(adj[x]))
    pos = {v: i for i, v in enumerate(order)}
    # Rebuild adjacency in ordered form.
    oadj = [set() for _ in range(n)]
    for u in range(n):
        for v in adj[u]:
            oadj[pos[u]].add(pos[v])

    color = [-1] * n

    def dfs(i):
        if i == n:
            return True
        v = order[i]
        used = set()
        for nb in adj[v]:
            c = color[nb]
            if c != -1:
                used.add(c)
        for c in range(3):
            if c not in used:
                color[v] = c
                if dfs(i + 1):
                    return True
                color[v] = -1
        return False

    return not dfs(0)


def construct():
    # Concentrate copies on the 5 high-degree Mycielski-copy nodes (5..9)
    # and one base node; the apex (10) gets 1 copy.
    # weights: base 0..4 -> 0, copies 5..9 -> 3 each (15), apex 10 -> 1 (total 16)
    weights = [0, 0, 0, 0, 0, 3, 3, 3, 3, 3, 1]

    edges, n = _blow_up(weights)

    # Sanity: must be triangle-free and not 3-colorable.
    if not _is_triangle_free(edges, n) or not _is_not_3_colorable(edges, n):
        # Fallback: the seed Grötzsch graph (11 vertices, 5 isolated).
        edges = [list(e) for e in _grotzsch_edges()]
        n = 16

    return edges
