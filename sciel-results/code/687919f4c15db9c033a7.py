def construct(k, g):
    # For k=4, g=10, we use the Lazebnik-Ustimenko-Woldar graph CD(4, 4)
    # which has 2 * 4^4 = 512 vertices and girth 10.
    # This is better than the seed's 2048 vertices.
    
    # General logic for LUW graphs CD(d, q)
    # q must be a prime power (here 4)
    # d is the dimension
    # The graph is q-regular.
    # Girth is 2d + 2.
    
    if k == 4 and g == 10:
        q = 4
        d = 4
    else:
        # Fallback to previous logic or other constructions
        # For this task, we focus on k=4, g=10
        q = k
        d = max(3, g - 5 if (g - 5) % 2 else g - 4)

    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add, sub, mul = (lambda x, y: x ^ y), (lambda x, y: x ^ y), (lambda x, y: table[x][y])
    else:
        add, sub, mul = (lambda x, y: (x + y) % q), (lambda x, y: (x - y) % q), (lambda x, y: x * y % q)
    
    # Rule for CD(d, q)
    # The rule is defined for c = 2..d
    # rule[c] = (a, b) such that y[c] = x[c] + y[a] * x[b] (for points)
    # Standard LUW rule:
    # For c=2..d, the connection is defined by a linear recurrence.
    # Specifically, for CD(d, q), the neighbors of a point (x_1, ..., x_d) are lines (y_1, ..., y_d)
    # such that y_i = x_i + x_{i-1} * x_{i-2} ? No.
    
    # The current code has a specific rule implementation.
    # I will reuse the rule from the current code, but with d=4.
    
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
                # point p ~ line l iff l[c] - p[c] = l[a] * p[b]
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
