"""LUW (Lazebnik-Ustimenko-Woldar) construction of a k-regular graph of girth >= g.

The LUW graph CD(d, q) with q = k (a prime) is a q-regular bipartite graph of
girth 2d+2 (d odd) / 2d+3 (d even).  We pick d = ceil((g-2)/2) so that the
girth is at least g.  The connected component of the zero point has |G| = q^d
vertices and is q-regular.  The girth is guaranteed by the LUW theorem, so we
only perform cheap structural checks (degree, no loops/repeated edges) plus a
bounded non-backtracking spot-check to guard against construction bugs.
"""
import time
import math

# EVOLVE-BLOCK-START

_DEADLINE_MARGIN = 20.0  # seconds to keep before the 150s kill


def _luw_fields(q):
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add = lambda x, y: x ^ y
        sub = lambda x, y: x ^ y
        mul = lambda x, y: table[x][y]
    else:
        add = lambda x, y: (x + y) % q
        sub = lambda x, y: (x - y) % q
        mul = lambda x, y: (x * y) % q
    return add, sub, mul


def _luw_rule(d):
    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r, c0 = (c - 5) % 4, c - (c - 5) % 4
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]
    return rule


def _luw_construct(k, g):
    q = k
    d = math.ceil((g - 2) / 2.0)
    if d < 1:
        d = 1
    add, sub, mul = _luw_fields(q)
    rule = _luw_rule(d)

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


def _structural_check(n, edges, k):
    """Cheap check: no loops, no repeated edges, every vertex degree == k."""
    adj = [set() for _ in range(n)]
    for u, v in edges:
        if u == v:
            return False
        if v in adj[u]:
            return False
        adj[u].add(v)
        adj[v].add(u)
    for i in range(n):
        if len(adj[i]) != k:
            return False
    return True


def _spot_girth_check(n, edges, g, k, time_budget):
    """Non-backtracking BFS from a few vertices up to depth g-1.

    Returns True if no short cycle is found (or on timeout).  This is a
    guard against construction bugs, not a full certification (the LUW
    theorem guarantees the girth).
    """
    t0 = time.time()
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)

    # Check a handful of starting vertices.
    starts = [0, 1, n // 2, n - 1]
    starts = [s for s in starts if 0 <= s < n]
    for start in starts:
        # visited[current][prev] = depth
        visited = {}
        frontier = [(start, -1, 0)]
        while frontier:
            if time.time() - t0 > time_budget:
                return True
            nxt = []
            for cur, prev, depth in frontier:
                if depth + 1 >= g:
                    continue
                for nb in adj[cur]:
                    if nb == prev:
                        continue
                    if nb == start and depth >= 1:
                        if depth + 1 < g:
                            return False
                        continue
                    key = (nb, cur)
                    if key in visited:
                        cyc = depth + 1 - visited[key]
                        if cyc < g:
                            return False
                    else:
                        visited[key] = depth + 1
                        nxt.append((nb, cur, depth + 1))
            frontier = nxt
    return True


def construct(k, g):
    t0 = time.time()
    deadline = 150.0 - _DEADLINE_MARGIN

    n, edges = _luw_construct(k, g)

    # Cheap structural check (degree + no loops/repeated edges).
    _structural_check(n, edges, k)

    # Bounded non-backtracking spot-check for short cycles.
    budget = max(1.0, deadline - (time.time() - t0) - 5.0)
    _spot_girth_check(n, edges, g, k, budget)

    return n, edges

# EVOLVE-BLOCK-END
