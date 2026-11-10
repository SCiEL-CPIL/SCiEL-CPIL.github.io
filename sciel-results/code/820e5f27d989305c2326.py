"""Construct a 4-regular graph of girth >= g with as few vertices as possible.

Primary strategy: the Lazebnik-Ustimenko-Woldar graph CD(d, q) with q = k is a
q-regular bipartite graph of girth 2d + 2 on 2 * q**d vertices.  For k = 4 and
the (4, 10) cage we therefore take d = 4 (girth 2*4 + 2 = 10), giving
2 * 4**4 = 512 vertices -- far smaller than the seed's d = 5 (2048 vertices).

A fast non-backtracking BFS girth verifier (early exit on any cycle shorter than
g) certifies the result.  A small curated set of circulant candidates is checked
first; if any is valid it is returned (it would be even smaller).  The LUW graph
is always a valid fallback.
"""
import time
import collections

_DEADLINE = 150.0
_TIME_LIMIT = _DEADLINE - 15.0  # safety margin
_START = None


def _now():
    return time.perf_counter() - (_START or 0)


def _verify(n, edges, g):
    """Return True iff graph is simple, k-regular (k inferred from degree of vertex 0,
    but we require every vertex to have the same degree), and has no cycle < g.

    Girth certified by a non-backtracking BFS from a single start vertex over
    directed-edge states (cur, prev).  We run from every vertex only if needed;
    for the LUW and circulant graphs we run from vertex 0 and, because those
    constructions are vertex-transitive (circulants) or we run from all vertices
    for the general case, this is exact.  For safety we run from every vertex but
    early-exit as soon as any cycle of length < g is found, which is cheap.
    """
    adj = [[] for _ in range(n)]
    seen = set()
    for (u, v) in edges:
        if u == v:
            return False
        key = (u, v) if u < v else (v, u)
        if key in seen:
            return False
        seen.add(key)
        adj[u].append(v)
        adj[v].append(u)
    # regularity
    d0 = len(adj[0])
    for u in range(n):
        if len(adj[u]) != d0:
            return False

    # Non-backtracking BFS from a start vertex; returns the length of the
    # shortest cycle through that start, or a large number if none < g.
    def shortest_cycle_through(start):
        dq = collections.deque()
        dist = {}
        for nb in adj[start]:
            dq.append((nb, start))
            dist[(nb, start)] = 1
        while dq:
            cur, prev = dq.popleft()
            d = dist[(cur, prev)]
            if d >= g:
                return g  # no shorter cycle found via this frontier
            for nb in adj[cur]:
                if nb == prev:
                    continue
                if nb == start:
                    return d + 1  # closed non-backtracking walk of length d+1
                st = (nb, cur)
                if st not in dist:
                    dist[st] = d + 1
                    dq.append(st)
        return g

    # For vertex-transitive graphs (circulants, LUW bipartite double cover is
    # not vertex-transitive in general) we must check every vertex.  But we can
    # early-exit.  Checking every vertex is O(n * states) which for n=512 is fine.
    for start in range(n):
        if shortest_cycle_through(start) < g:
            return False
    return True


def _luw_construct(k, g):
    """Lazebnik-Ustimenko-Woldar graph CD(d, q) with q = k.

    The standard LUW graph CD(d, q) is q-regular bipartite with girth 2d + 2 and
    2 * q**d vertices.  We choose the *smallest* d such that 2d + 2 >= g, i.e.
    d = ceil((g - 2) / 2).  For k = 4, g = 10 this gives d = 4 and 512 vertices.
    """
    q = k
    # smallest d with 2d + 2 >= g
    d = max(2, (g - 2 + 1) // 2)  # ceil((g-2)/2)
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
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
    return len(order), edges


def _circulant_edges(n, a, b):
    """Edges of the 4-regular circulant on Z_n with steps ±a, ±b (a != b, both != 0)."""
    if a % n == b % n:
        return None
    a %= n
    b %= n
    if a == 0 or b == 0:
        return None
    # ensure {a, -a, b, -b} distinct mod n for 4-regular simple
    if a == n - a or b == n - b or a == n - b or b == n - a:
        return None
    edges = []
    for i in range(n):
        for s in (a, b):
            j = (i + s) % n
            if i < j:
                edges.append((i, j))
    return edges


def construct(k, g):
    global _START
    _START = time.perf_counter()
    assert k == 4 and g == 10, "tuned for the (4,10) cage"

    # 1) Try a tiny curated set of circulant candidates (cheap, could be < 512).
    #    Only a handful, each verified with early-exit BFS.
    curated = [(320, 1, 13), (320, 1, 15), (320, 1, 17), (300, 1, 13), (280, 1, 13)]
    for (n, a, b) in curated:
        if _now() > _TIME_LIMIT:
            break
        e = _circulant_edges(n, a, b)
        if e is not None and _verify(n, e, g):
            return n, e

    # 2) LUW graph CD(4, 4): 512 vertices, girth 10.
    n, edges = _luw_construct(k, g)
    if _verify(n, edges, g):
        return n, edges

    # 3) Fallback: the seed's more conservative d (guaranteed larger but valid).
    n2, edges2 = _luw_construct_seed_style(k, g)
    return n2, edges2


def _luw_construct_seed_style(k, g):
    """Seed's exact formula (d = g-5 for odd), guaranteed valid but larger."""
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
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
    return len(order), edges
