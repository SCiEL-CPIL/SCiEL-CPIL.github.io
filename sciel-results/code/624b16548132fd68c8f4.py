import time

# EVOLVE-BLOCK-START
def _luw_graph(q, d):
    """Return (n, edges) for the Lazebnik-Ustimenko-Woldar CD(d,q) graph,
    restricted to the connected component of the zero point.
    q-regular, bipartite, girth 2d+2 (for prime q)."""
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add = lambda x, y: x ^ y
        sub = lambda x, y: x ^ y
        mul = lambda x, y: table[x][y]
    else:
        add = lambda x, y: (x + y) % q
        sub = lambda x, y: (x - y) % q
        mul = lambda x, y: x * y % q

    # rule[c] = (a, b) for c = 2..d, point~line iff l[c]-p[c] = l[a]*p[b]
    rule = {}
    for c in range(2, d + 1):
        r = (c - 5) % 4
        c0 = c - r
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]

    def neighbours(vertex):
        side, x = vertex
        for first in range(q):
            y = [None, first] + [0] * (d - 1)
            for c in range(2, d + 1):
                a, b = rule[c]
                if side == 0:
                    y[c] = add(x[c], mul(y[a], x[b]))
                else:
                    y[c] = sub(x[c], mul(x[a], y[b]))
            yield (1 - side), tuple(y)

    start = (0, (None,) + (0,) * d)
    index = {start: 0}
    order = [start]
    edges = []
    for vertex in order:
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
    return len(order), edges


def _girth_ok(n, edges, g, deadline=None):
    """Return True iff the graph has no cycle of length < g, checked exactly
    by non-backtracking walks from every vertex. deadline: absolute time after
    which we stop and return the best-so-far (True if no short cycle seen yet)."""
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    # degree check
    for u in range(n):
        if len(adj[u]) != 4:
            return False
    # non-backtracking: state (current, prev)
    # walk length L means L edges; a cycle of length c < g appears as a
    # non-backtracking closed walk. We check walks of length up to g-1.
    # Use BFS per starting vertex with states (cur, prev).
    for s in range(n):
        # layer 0: just at s
        # We want to detect a closed non-backtracking walk of length in [4, g-1]
        # (bipartite so even; shortest possible cycle >=4).
        # frontier: dict (cur, prev) -> None, start with prev = -1.
        frontier = [(s, -1)]
        for L in range(1, g):
            nxt = set()
            for cur, prev in frontier:
                for w in adj[cur]:
                    if w == prev:
                        continue
                    # if w == s and L >= 4 and L < g -> cycle found
                    if w == s and L >= 4 and L < g:
                        return False
                    nxt.add((w, cur))
            frontier = list(nxt)
            if deadline is not None and time.time() > deadline:
                return True  # no short cycle found yet
    return True


def _try_luw(k, g, deadline):
    """Return (n, edges) for the smallest-d LUW graph with girth >= g, or None."""
    q = k
    # d must satisfy 2d+2 >= g ideally; try d from floor((g-2)/2) upward.
    # But for q=4 (not prime) girth may be less, so verify.
    d_min = (g - 2) // 2
    if d_min < 3:
        d_min = 3
    for d in range(d_min, d_min + 3):
        if time.time() > deadline:
            break
        n, edges = _luw_graph(q, d)
        # degree and girth check
        if _girth_ok(n, edges, g, deadline):
            return n, edges
    return None


def _fallback_luw(k, g):
    """Conservative LUW construction (d = g-5 or g-4, as in seed)."""
    q = k
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    return _luw_graph(q, d)


def construct(k, g):
    deadline = time.time() + 110.0  # 10s safety margin
    # First try to find a smaller-d LUW graph that genuinely meets girth.
    result = _try_luw(k, g, deadline)
    if result is not None:
        return result
    # Fallback: conservative construction.
    return _fallback_luw(k, g)
# EVOLVE-BLOCK-END
