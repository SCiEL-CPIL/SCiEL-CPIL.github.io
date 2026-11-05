"""Improved program for the suite task; uses minimal LUW parameters."""
import time

# EVOLVE-BLOCK-START
def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # Lazebnik-Ustimenko-Woldar graph CD(d, q) with q = k.
    # The girth of CD(d, q) is 2d + 1.
    # We require girth >= g, so 2d + 1 >= g => d >= (g-1)/2.
    # We choose the smallest integer d to minimize vertices n = 2 * q^d.
    q = k
    d = max(1, (g - 1) // 2)
    
    # If q is 4, we need special arithmetic (GF(4) is not a prime field, but a field of order 4)
    # However, for k=5, q=5 is prime. The seed code handled q=4.
    # We keep the general structure.
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add, sub, mul = (lambda x, y: x ^ y), (lambda x, y: x ^ y), (lambda x, y: table[x][y])
    else:
        add, sub, mul = (lambda x, y: (x + y) % q), (lambda x, y: (x - y) % q), (lambda x, y: x * y % q)
        
    # Define the rules for the LUW graph.
    # For small d, the rules are standard.
    # The seed code provided a pattern for c > 4.
    rule = {}
    if d >= 2: rule[2] = (1, 1)
    if d >= 3: rule[3] = (2, 1)
    if d >= 4: rule[4] = (1, 2)
    
    for c in range(5, d + 1):
        r, c0 = (c - 5) % 4, c - (c - 5) % 4
        # The seed code used this list. We replicate it.
        # Note: The indices in the list correspond to the rule (a, b)
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]

    def neighbours(vertex):
        side, x = vertex
        for first in range(q):
            # y[0] is unused/None, y[1] is 'first', y[2..d] are calculated
            # x is a tuple of length d+1 (indices 0..d), x[0] is None/unused
            y = [None, first] + [0] * (d - 1)
            for c in range(2, d + 1):
                a, b = rule[c]
                if side == 0:
                    # point p (side 0) ~ line l (side 1)
                    # l[c] - p[c] = l[a] * p[b]
                    # y is the line, x is the point
                    # y[c] = x[c] + y[a] * x[b] ?
                    # Wait, the seed code had:
                    # y[c] = add(x[c], mul(y[a], x[b])) if side == 0
                    # This implies y[a] is already known?
                    # In the LUW construction, the coordinates are computed sequentially.
                    # y[1] is set.
                    # For c=2, a,b are from rule[2].
                    # If a=1, y[1] is known.
                    # So y[c] depends on previous y's.
                    y[c] = add(x[c], mul(y[a], x[b]))
                else:
                    # side == 1 (line), x is line, y is point
                    # p[c] - l[c] = l[a] * p[b] ??
                    # Seed code: y[c] = sub(x[c], mul(x[a], y[b]))
                    # Here x is the line (current vertex), y is the point (neighbor).
                    # So y[b] is a coordinate of the neighbor point.
                    # If b < c, y[b] is known.
                    y[c] = sub(x[c], mul(x[a], y[b]))
            yield 1 - side, tuple(y)

    start = (0, (None,) + (0,) * d)
    index, order, edges = {start: 0}, [start], []
    
    # BFS to find connected component
    for i, vertex in enumerate(order):
        if time.time() > deadline:
            break
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            # Only add edge from point to line to avoid duplicates
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
                
    return len(order), edges
# EVOLVE-BLOCK-END
