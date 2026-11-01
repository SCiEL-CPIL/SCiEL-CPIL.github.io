"""Lazebnik-Ustimenko-Woldar explicit q-regular graph (zero component), girth >= g.

construct(k, g) -> (n, edges). For this task k=3, g=16.
We use the L-U-W graph CD(d, q) with q = k and d chosen odd so that the
girth of the zero-point component is at least d + 5 >= g.
"""
import time

_DEADLINE = 150.0
_START = time.time()


def _time_left():
    return _DEADLINE - (time.time() - _START)


def construct(k, g):
    q = k
    # Need girth >= g. L-U-W zero component girth = d + 5 (d odd).
    # Choose smallest odd d with d + 5 >= g.
    d = g - 5
    if d % 2 == 0:
        d += 1
    if d < 3:
        d = 3 if g <= 8 else 5

    # Field GF(q). q is prime here (k=3). Keep generic for prime q.
    def add(x, y):
        return (x + y) % q

    def sub(x, y):
        return (x - y) % q

    def mul(x, y):
        return (x * y) % q

    # Rule: for c in 2..d, the relation between point p and line l is
    #   l[c] - p[c] = l[a] * p[b]   (point side)
    # The L-U-W rule (d odd):
    rule = {}
    rule[2] = (1, 1)
    rule[3] = (2, 1)
    rule[4] = (1, 2)
    for c in range(5, d + 1):
        r = (c - 5) % 4
        c0 = c - r
        # pattern of (a,b)
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
    index = {start: 0}
    order = [start]
    edges = []
    i = 0
    while i < len(order):
        if _time_left() < 5.0:
            break
        vertex = order[i]
        i += 1
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))

    n = len(order)
    return n, edges
