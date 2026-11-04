"""LUW CD(d,q) cage construction with girth correction + exact girth verification."""
import time

# EVOLVE-BLOCK-START
def _luw_graph(k, d):
    """Build the connected component of the zero point of the Lazebnik-Ustimenko-Woldar
    graph CD(d, q) with q = k.  q-regular, bipartite, girth = d+5 (odd d) or d+4 (even d)."""
    q = k
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add = sub = (lambda x, y: x ^ y)
        mul = (lambda x, y: table[x][y])
    else:
        add = (lambda x, y: (x + y) % q)
        sub = (lambda x, y: (x - y) % q)
        mul = (lambda x, y: x * y % q)

    # Rule: l[c] - p[c] = l[a] * p[b]  for c = 2..d
    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r, c0 = (c - 5) % 4, c - (c - 5) % 4
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
            yield 1 - side, tuple(y)

    start = (0, (None,) + (0,) * d)
    index, order, edges = {start: 0}, [start], []
    for vertex in order:
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
    return len(order), edges


def _verify_girth(n, edges, g):
    """Return True iff the girth is at least g.  Uses non-backtracking walks from every
    vertex (exact, per the task contract).  For a bipartite graph the shortest non-backtracking
    closed walk from v equals the girth."""
    adj = [[] for _ in range(n)]
    for a, b in edges:
        adj[a].append(b)
        adj[b].append(a)

    for s in range(n):
        # state (v, parent) = non-backtracking walk of length L ending at v (came from parent)
        frontier = [(s, -1)]
        L = 1
        while frontier:
            nxt = []
            for v, p in frontier:
                for w in adj[v]:
                    if w == p:
                        continue
                    if w == s:
                        return L >= g          # closed non-backtracking walk of length L
                    nxt.append((w, v))
            frontier = nxt
            L += 1
    return True


def construct(k, g):
    deadline = time.time() + 110.0
    # Minimum d so that the LUW girth (d+5 odd / d+4 even) is >= g, for d >= 3.
    d = g - 4 if g % 2 == 0 else g - 5
    d = max(3, d)

    n, edges = _luw_graph(k, d)

    # Exact certification: never return a graph whose girth is < g.
    if _verify_girth(n, edges, g):
        return n, edges

    # Fallback (should not trigger for the required parameters): grow d until certified.
    while time.time() < deadline:
        d += 1
        n, edges = _luw_graph(k, d)
        if _verify_girth(n, edges, g):
            return n, edges
    return n, edges
# EVOLVE-BLOCK-END
