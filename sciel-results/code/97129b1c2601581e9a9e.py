def _luw_seed(k, g):
    """The seed Lazebnik-Ustimenko-Woldar CD(d, q) construction (valid)."""
    q = k
    # LUW CD(d, q) has girth 2d+1. We need girth >= g.
    # So 2d + 1 >= g => d >= (g-1)/2.
    d = (g - 1 + 1) // 2  # Ceiling of (g-1)/2
    if d < 2: d = 2 # Minimum d for LUW is usually 2 or 3? 
    # For d=2, girth 5. Vertices 2*q^2.
    # For d=3, girth 7. Vertices 2*q^3.
    # For d=4, girth 9. Vertices 2*q^4.
    
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add, sub, mul = (lambda x, y: x ^ y), (lambda x, y: x ^ y), (lambda x, y: table[x][y])
    else:
        add, sub, mul = (lambda x, y: (x + y) % q), (lambda x, y: (x - y) % q), (lambda x, y: x * y % q)
    
    # Rules for LUW
    # c=2: (1,1)
    # c=3: (2,1)
    # c=4: (1,2)
    # c=5: (1,4)? Let's check the pattern in the seed.
    # Seed: rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]
    # where r = (c-5)%4, c0 = c - r.
    # For c=5: r=0, c0=5. rule[5] = (1, 3).
    # For c=6: r=1, c0=6. rule[6] = (5, 1).
    # For c=7: r=2, c0=7. rule[7] = (7, 1).
    # For c=8: r=3, c0=8. rule[8] = (1, 9).
    
    # The seed code initializes rule for 2,3,4 and computes 5..d.
    # This is correct.
    
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
