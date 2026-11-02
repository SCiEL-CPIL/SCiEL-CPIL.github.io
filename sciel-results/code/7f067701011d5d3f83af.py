"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct(k, g):
    # Lazebnik-Ustimenko-Woldar graph CD(d, q) with q = k.
    # Girth is 2d + 2. We need 2d + 2 >= g, so d >= (g - 2) / 2.
    # We choose the smallest integer d satisfying this.
    q = k
    d = max(1, (g - 2 + 1) // 2)
    
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add, sub, mul = (lambda x, y: x ^ y), (lambda x, y: x ^ y), (lambda x, y: table[x][y])
    else:
        add, sub, mul = (lambda x, y: (x + y) % q), (lambda x, y: (x - y) % q), (lambda x, y: x * y % q)
    
    # Define the rule for the LUW graph
    # For c=2..d, the rule is a pair (a, b) such that the connection is defined.
    # Standard LUW rules:
    # c=2: (1, 1)
    # c=3: (2, 1)
    # c=4: (1, 2)
    # For c >= 5, the rule cycles or follows a pattern.
    # The seed code used a specific pattern. We keep it.
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
                # point p ~ line l iff l[c] - p[c] = l[a] * p[b] for c = 2..d
                # side 0 is point, side 1 is line.
                # If side==0 (point x), we find line y.
                # y[c] = x[c] + y[a] * x[b]  => y[c] = x[c] + y[a] * x[b]
                # Wait, the equation is l[c] - p[c] = l[a] * p[b].
                # So y[c] - x[c] = y[a] * x[b] => y[c] = x[c] + y[a] * x[b].
                # If side==1 (line x), we find point y.
                # x[c] - y[c] = x[a] * y[b] => y[c] = x[c] - x[a] * y[b].
                
                if side == 0:
                    y[c] = add(x[c], mul(y[a], x[b]))
                else:
                    y[c] = sub(x[c], mul(x[a], y[b]))
            yield 1 - side, tuple(y)

    start = (0, (None,) + (0,) * d)
    index, order, edges = {start: 0}, [start], []
    
    import time
    start_time = time.time()
    
    for vertex in order:
        if time.time() - start_time > 100:
            break
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
                
    return len(order), edges
# EVOLVE-BLOCK-END
