"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct(k, g):
    # Lazebnik-Ustimenko-Woldar graph CD(d, q) with q = k: an explicit q-regular bipartite graph
    # whose girth is at least d + 5. Points and lines are vectors over GF(q).
    # We choose the minimal d such that d + 5 >= g.
    q = k
    # Minimal d for girth >= g
    d = max(1, g - 5)
    
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add, sub, mul = (lambda x, y: x ^ y), (lambda x, y: x ^ y), (lambda x, y: table[x][y])
    else:
        add, sub, mul = (lambda x, y: (x + y) % q), (lambda x, y: (x - y) % q), (lambda x, y: x * y % q)
        
    # Define the recurrence rules for the LUW graph
    # The rule for c depends on c mod 4
    rule = {}
    # Base cases
    rule[2] = (1, 1)
    rule[3] = (2, 1)
    rule[4] = (1, 2)
    
    for c in range(5, d + 1):
        r = (c - 5) % 4
        c0 = c - r
        # The standard LUW rules
        if r == 0:
            rule[c] = (1, c0 - 2)
        elif r == 1:
            rule[c] = (c0 - 1, 1)
        elif r == 2:
            rule[c] = (c0, 1)
        else: # r == 3
            rule[c] = (1, c0 + 1)

    def neighbours(vertex):
        side, x = vertex
        for first in range(q):
            # x is a tuple of length d+1 (indices 0..d)
            # y[0] is None, y[1] is 'first', y[2..d] are calculated
            y = [None, first] + [0] * (d - 1)
            for c in range(2, d + 1):
                a, b = rule[c]
                # Point p ~ Line l iff l[c] - p[c] = l[a] * p[b] (for points side=0)
                # Or p[c] - l[c] = p[a] * l[b] (for lines side=1)
                # In the original code:
                # y[c] = add(x[c], mul(y[a], x[b])) if side == 0 else sub(x[c], mul(x[a], y[b]))
                # Note: x is the current vertex, y is the neighbor.
                # If side==0 (point), we compute line y.
                # Relation: y[c] - x[c] = y[a] * x[b]  => y[c] = x[c] + y[a] * x[b]
                # If side==1 (line), we compute point y.
                # Relation: x[c] - y[c] = x[a] * y[b]  => y[c] = x[c] - x[a] * y[b]
                
                if side == 0:
                    y[c] = add(x[c], mul(y[a], x[b]))
                else:
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
# EVOLVE-BLOCK-END
