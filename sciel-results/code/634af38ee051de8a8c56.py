"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct(k, g):
    # Lazebnik-Ustimenko-Woldar graph CD(d, q) with q = k: an explicit q-regular bipartite graph
    # whose girth is at least 2d + 2. Points and lines are vectors over GF(q) (q prime,
    # or q = 4); the connected component of the zero point is returned.
    q = k
    # We need girth >= g.
    # Girth of LUW is 2d + 2.
    # So 2d + 2 >= g => 2d >= g - 2 => d >= (g - 2) / 2.
    # d must be an integer.
    d = (g - 2) // 2
    if d < 1:
        d = 1
        
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add, sub, mul = (lambda x, y: x ^ y), (lambda x, y: x ^ y), (lambda x, y: table[x][y])
    else:
        add, sub, mul = (lambda x, y: (x + y) % q), (lambda x, y: (x - y) % q), (lambda x, y: x * y % q)
        
    # The rule for the LUW graph construction
    # For d=1, no rules needed for c in 2..1 (empty)
    # For d=2, rule[2]
    # For d=3, rule[2], rule[3]
    # For d=4, rule[2], rule[3], rule[4]
    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2), 5: (1, 3), 6: (2, 3)}
    # Extend rule if d > 6?
    for c in range(5, d + 1):
        if c not in rule:
             r, c0 = (c - 5) % 4, c - (c - 5) % 4
             rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]

    def neighbours(vertex):
        side, x = vertex
        for first in range(q):
            y = [None, first] + [0] * (d - 1)
            for c in range(2, d + 1):
                a, b = rule[c]
                # point p ~ line l  iff  l[c] - p[c] = l[a] * p[b] for c = 2..d
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
# EVOLVE-BLOCK-END
