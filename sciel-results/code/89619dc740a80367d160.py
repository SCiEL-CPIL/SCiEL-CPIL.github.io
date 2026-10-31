import time

def construct(k, g):
    start_time = time.time()
    deadline = 150.0 - 5.0

    if (g - 5) % 2 == 1 and g - 5 >= 3:
        d = g - 5
    elif (g - 4) % 2 == 0 and g - 4 >= 2:
        d = g - 4
    else:
        d = max(3, g - 5)
        if d % 2 == 0:
            d += 1

    q = k

    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        def add(x, y): return x ^ y
        def sub(x, y): return x ^ y
        def mul(x, y): return table[x][y]
    else:
        def add(x, y): return (x + y) % q
        def sub(x, y): return (x - y) % q
        def mul(x, y): return (x * y) % q

    rule = {}
    rule[2] = (1, 1)
    rule[3] = (2, 1)
    rule[4] = (1, 2)
    for c in range(5, d + 1):
        r = (c - 5) % 4
        c0 = c - r
        if r == 0:
            rule[c] = (1, c0 - 2)
        elif r == 1:
            rule[c] = (c0 - 1, 1)
        elif r == 2:
            rule[c] = (c0, 1)
        elif r == 3:
            rule[c] = (1, c0 + 1)

    def get_neighbours(vertex):
        side, x = vertex
        neighbours = []
        for first in range(q):
            y = [None] + [0] * d
            y[1] = first
            for c in range(2, d + 1):
                a, b = rule[c]
                if side == 0:
                    y[c] = add(x[c], mul(y[a], x[b]))
                else:
                    y[c] = sub(x[c], mul(x[a], y[b]))
            neighbours.append((1 - side, tuple(y[1:])))
        return neighbours

    start = (0, tuple([0] * d))
    index = {start: 0}
    order = [start]
    edges = []

    for i in range(len(order)):
        if time.time() > deadline:
            break
        vertex = order[i]
        for other in get_neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))

    n = len(order)
    return n, edges
