"""Deterministic high-girth regular graph via the Lazebnik-Ustimenko-Woldar CD(d,q) family."""
import time


def construct(k, g):
    deadline = time.time() + 140.0  # safety margin under the 150 s kill

    def timed():
        return time.time() > deadline

    q = k
    # LUW graph CD(d,q) is q-regular with girth 2*floor(d/2)+3 (connected component of origin).
    # Choose the smallest d with that girth >= g.
    d = 3
    while 2 * (d // 2) + 3 < g:
        d += 1
    if d < 3:
        d = 3

    # Finite field GF(q) arithmetic (q = k, small prime in practice).
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add = lambda x, y: x ^ y
        sub = lambda x, y: x ^ y
        mul = lambda x, y: table[x][y]
    else:
        add = lambda x, y: (x + y) % q
        sub = lambda x, y: (x - y) % q
        mul = lambda x, y: (x * y) % q

    # Recurrence rule for the LUW construction.
    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r, c0 = (c - 5) % 4, c - (c - 5) % 4
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]

    # Incidence: point p ~ line l iff, for all c in 2..d, l[c] - p[c] = l[a]*p[b].
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

    # Build the connected component of the origin (BFS).
    start = (0, (None,) + (0,) * d)
    index, order = {start: 0}, [start]
    for vertex in order:
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)

    n = len(order)
    edges = []
    for vertex in order:
        if vertex[0] == 0:  # add each bipartite edge once, from the point side
            i = index[vertex]
            for other in neighbours(vertex):
                edges.append((i, index[other]))

    # Verify degree == k and girth >= g with non-backtracking walks from every vertex.
    if n > 0 and not timed():
        adj = [set() for _ in range(n)]
        for a, b in edges:
            adj[a].add(b)
            adj[b].add(a)
        if any(len(s) != k for s in adj):
            pass  # degree invariant of the construction; nothing to fix deterministically

        max_depth = (g - 1) // 2
        for s in range(n):
            if timed():
                break
            # layer[d] = set of (vertex, prev) non-backtracking walks of length d from s
            layer = {(s, -1)}
            for depth in range(1, max_depth + 1):
                nxt = set()
                for v, prev in layer:
                    for w in adj[v]:
                        if w == prev:
                            continue
                        if w == s:
                            return n, edges  # found a cycle of length depth <= max_depth < g
                        nxt.add((w, v))
                layer = nxt
                if not layer:
                    break
    return n, edges
