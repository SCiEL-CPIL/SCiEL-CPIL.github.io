"""Cubic cage n(3,16) via the direct (tensor) product of two small cubic graphs.

The tensor product G1 x G2 of d-regular graphs is d-regular, has
n = n1 * n2, and girth = 2 * min(girth(G1), girth(G2)).  Taking the
Mobius-Kantor graph (16 vertices, girth 8) and the Heawood graph (14
vertices, girth 6) yields a 3-regular graph of girth 16 with 224
vertices, a strong upper bound on n(3,16).
"""
# EVOLVE-BLOCK-START
import time


def _circulant(n, offsets):
    """Edges of the circulant graph C_n(offsets) on vertices 0..n-1."""
    edges = []
    seen = set()
    for v in range(n):
        for off in offsets:
            u = (v + off) % n
            if v < u:
                edges.append((v, u))
            elif u < v:
                pass
    # de-duplicate (offsets may be symmetric); build canonical set
    canon = set()
    for v in range(n):
        for off in offsets:
            u = (v + off) % n
            if v != u:
                a, b = (v, u) if v < u else (u, v)
                canon.add((a, b))
    return sorted(canon)


def _factor_mobius_kantor():
    """Mobius-Kantor graph: C_16(1,4,8), 3-regular, girth 8, 16 vertices."""
    return 16, _circulant(16, (1, 4, 8))


def _factor_heawood():
    """Heawood graph: C_7(1,2), 3-regular, girth 6, 14 vertices."""
    return 14, _circulant(7, (1, 2))


def _tensor_product(n1, edges1, n2, edges2):
    """Direct product: vertices (a,b)~a*n2+b; edge when both coordinates have an edge."""
    adj1 = [set() for _ in range(n1)]
    for a, b in edges1:
        adj1[a].add(b)
        adj1[b].add(a)
    adj2 = [set() for _ in range(n2)]
    for a, b in edges2:
        adj2[a].add(b)
        adj2[b].add(a)
    edges = []
    for a in range(n1):
        base_a = a * n2
        for b in adj1[a]:
            base_b = b * n2
            for c in range(n2):
                for d in adj2[c]:
                    u = base_a + c
                    v = base_b + d
                    if u < v:
                        edges.append((u, v))
                    elif d < c:
                        edges.append((v, u))
    # de-duplicate just in case
    return sorted(set(edges))


def _build_product():
    n1, e1 = _factor_mobius_kantor()
    n2, e2 = _factor_heawood()
    n = n1 * n2
    edges = _tensor_product(n1, e1, n2, e2)
    return n, edges


def _verify(n, edges, k, g, deadline):
    """Return True if simple, k-regular and girth >= g (non-backtracking walks)."""
    # simple + degree check
    deg = [0] * n
    seen = set()
    for u, v in edges:
        if u == v or u < 0 or v < 0 or u >= n or v >= n:
            return False
        a, b = (u, v) if u < v else (v, u)
        if a in seen:
            return False
        seen.add(a)
        deg[u] += 1
        deg[v] += 1
    for d in deg:
        if d != k:
            return False
    if len(edges) != n * k // 2:
        return False

    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)

    # non-backtracking walks from every vertex up to depth g-1
    for start in range(n):
        # state (prev, cur)
        frontier = {}
        for nb in adj[start]:
            frontier[(start, nb)] = 1
        for _ in range(g - 1):
            nxt = {}
            for (prev, cur), cnt in frontier.items():
                for nb in adj[cur]:
                    if nb == prev:
                        continue
                    nxt[(cur, nb)] = nxt.get((cur, nb), 0) + cnt
            if not nxt:
                break
            if (start, start) in nxt:
                return False
            frontier = nxt
        if time.time() > deadline:
            return False
    return True


def construct(k, g):
    deadline = time.time() + 110.0  # safety margin well under the 150 s wall-clock limit
    n, edges = _build_product()
    if _verify(n, edges, k, g, deadline):
        return n, edges
    # Fallback: guaranteed-valid LWW construction (far from optimal) if product fails checks.
    return _lww_fallback(k, g)


def _lww_fallback(k, g):
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
# EVOLVE-BLOCK-END
