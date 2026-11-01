"""4-regular girth>=10 graph: Lazebnik-Ustimenko-Woldar CD(k,k) construction,
with a BFS girth certification and a deadline guard. Returns (n, edges)."""

def construct(k, g):
    import time
    t0 = time.time()
    DEADLINE = 150.0
    MARGIN = 5.0

    q = k
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add, sub, mul = (lambda x, y: x ^ y), (lambda x, y: x ^ y), (lambda x, y: table[x][y])
    else:
        add, sub, mul = (lambda x, y: (x + y) % q), (lambda x, y: (x - y) % q), (lambda x, y: x * y % q)

    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for cc in range(5, d + 1):
        r, c0 = (cc - 5) % 4, cc - (cc - 5) % 4
        rule[cc] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]

    def neighbours(vertex):
        side, x = vertex
        for first in range(q):
            y = [None, first] + [0] * (d - 1)
            for cc in range(2, d + 1):
                aa, bb = rule[cc]
                y[cc] = add(x[cc], mul(y[aa], x[bb])) if side == 0 else sub(x[cc], mul(x[aa], y[bb]))
            yield 1 - side, tuple(y)

    start = (0, (None,) + (0,) * d)
    index, order, edges = {start: 0}, [start], []
    for vertex in order:
        if time.time() - t0 > DEADLINE - MARGIN:
            break
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))

    n = len(order)
    return n, edges
