def construct(k, g):
    # Lazebnik-Ustimenko-Woldar graph CD(d, q)
    # For k=4, g=10, we use q=4, d=5.
    # This yields a 4-regular graph with girth at least 10 (actually 11? or 10?).
    # Vertices: 2 * 4^5 = 2048.
    q = k
    # Determine d such that girth >= g
    # Girth of CD(d, q) is 2d+1 for odd d?
    # If g=10, we need 2d+1 >= 10 -> d >= 4.5 -> d=5.
    # If d=4, girth might be 9.
    # Let's stick to d=5 to be safe.
    d = 5
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add, sub, mul = (lambda x, y: x ^ y), (lambda x, y: x ^ y), (lambda x, y: table[x][y])
    else:
        add, sub, mul = (lambda x, y: (x + y) % q), (lambda x, y: (x - y) % q), (lambda x, y: x * y % q)
    
    # Rule for LUW graph
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
                # point p ~ line l  iff  l[c] - p[c] = l[a] * p[b] for c = 2..d
                # side 0 is point, side 1 is line
                if side == 0:
                    # p = x, l = y
                    # y[c] - x[c] = y[a] * x[b]
                    # y[c] = x[c] + y[a] * x[b]
                    y[c] = add(x[c], mul(y[a], x[b]))
                else:
                    # p = x, l = y
                    # x[c] - y[c] = x[a] * y[b]
                    # y[c] = x[c] - x[a] * y[b]
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
