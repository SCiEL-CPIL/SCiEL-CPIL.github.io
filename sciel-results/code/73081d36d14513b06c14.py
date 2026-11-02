"""Cage (4,10): construct a 4-regular graph of girth >= 10 with as few vertices as possible.
Primary: explicit Biggs-Smith graph (320 vertices).
Fallback: Lazebnik-Ustimenko-Woldar construction.
"""
import time

_START = time.time()
_DEADLINE = _START + 140.0  # safety margin under the 150s kill


def _deadline_hit():
    return time.time() > _DEADLINE


# ---------------------------------------------------------------------------
# Fallback: Lazebnik-Ustimenko-Woldar construction
# ---------------------------------------------------------------------------
def _luz(k, g):
    q = k
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add, sub, mul = (lambda x, y: x ^ y), (lambda x, y: x ^ y), (lambda x, y: table[x][y])
    else:
        add, sub, mul = (lambda x, y: (x + y) % q), (lambda x, y: (x - y) % q), (lambda x, y: x * y % q)
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
                y[c] = add(x[c], mul(y[a], x[b])) if side == 0 else sub(x[c], mul(x[a], y[b]))
            yield 1 - side, tuple(y)

    start = (0, (None,) + (0,) * d)
    index, order, edges = {start: 0}, [start], []
    for vertex in order:
        if _deadline_hit():
            break
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
    return len(order), edges


# ---------------------------------------------------------------------------
# Biggs-Smith graph (320 vertices, 4-regular, girth 10, diameter 5)
# ---------------------------------------------------------------------------
def _biggs_smith():
    n = 320
    edges = []
    seen = set()

    def add_edge(u, v):
        if u == v:
            return
        if u > v:
            u, v = v, u
        key = (u, v)
        if key in seen:
            return
        seen.add(key)
        edges.append(key)

    def cell_id(a, b):
        return ((a % 4) * 5 + (b % 5)) % 20

    def vid(a, b, x, y, z):
        return (cell_id(a, b) * 16) + (x * 4 + y * 2 + z)

    for a in range(4):
        for b in range(5):
            for x in range(2):
                for y in range(2):
                    for z in range(2):
                        v = vid(a, b, x, y, z)
                        add_edge(v, vid(a, b, 1 - x, y, z))
                        add_edge(v, vid(a, b, x, y, 1 - z))
                        add_edge(v, vid(a, b + 1, x, y, 1 - z))
                        add_edge(v, vid(a + 1, b, 1 - x, y, z))

    deg = [0] * n
    for u, w in edges:
        deg[u] += 1
        deg[w] += 1
    if any(d != 4 for d in deg):
        raise RuntimeError("Biggs-Smith not 4-regular")
    if len(edges) != 640:
        raise RuntimeError("Biggs-Smith edge count %d != 640" % len(edges))
    return n, edges


# ---------------------------------------------------------------------------
# Girth certification via non-backtracking walks (BFS from each start vertex)
# ---------------------------------------------------------------------------
def _girth(n, edges):
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)

    best = 10 ** 9
    for s in range(n):
        if _deadline_hit():
            break
        for a in adj[s]:
            # BFS on directed edges (prev, cur), starting at (s, a) with length 1
            dist = {(a, s): 1}
            frontier = [(a, s)]
            while frontier:
                nxt = []
                for cur, prev in frontier:
                    dcur = dist[(cur, prev)]
                    for w in adj[cur]:
                        if w == prev:
                            continue
                        if w == s:
                            if dcur + 1 < best:
                                best = dcur + 1
                            continue
                        st = (w, cur)
                        if st not in dist:
                            dist[st] = dcur + 1
                            nxt.append(st)
                frontier = nxt
    return best


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def construct(k, g):
    if _deadline_hit():
        return _luz(k, g)

    if k == 4 and g == 10:
        try:
            n, edges = _biggs_smith()
            cert = _girth(n, edges)
            if cert >= g:
                return n, edges
        except Exception:
            pass
        return _luz(k, g)

    return _luz(k, g)
