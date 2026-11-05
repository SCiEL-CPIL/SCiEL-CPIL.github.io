"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct(k, g):
    # Lazebnik-Ustimenko-Woldar graph CD(d, q) with q = k: an explicit q-regular bipartite graph
    # whose girth is at least d + 5 for odd d. Points and lines are vectors over GF(q) (q prime,
    # or q = 4); the connected component of the zero point is returned. Valid, far from optimal.
    q = k
    # For girth g, we need d such that d + 5 >= g.
    # The LUW graph CD(d, q) has girth 2d + 2 if q is large enough?
    # Actually, the girth of CD(d, q) is 2d + 2 for d >= 3?
    # The seed uses d = g - 5 if (g-5) is odd, else g - 4.
    # For g=9, g-5=4 (even), so d = 9 - 4 = 5?
    # Wait, the seed code says:
    # d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    # If g=9, (9-5)=4, 4%2==0, so d = 9 - 4 = 5.
    # If d=5, n = 2 * 5^5 = 6250.
    # If d=4, girth is 2*4+2 = 10? Or 4+5=9?
    # The comment says "girth is at least d + 5 for odd d".
    # If d is even, girth is at least d + 4?
    # Let's check the rule.
    # The standard LUW graph CD(d, q) has girth 2d + 2.
    # If we want girth 9, we need 2d + 2 >= 9 => 2d >= 7 => d >= 3.5 => d = 4.
    # So d=4 should give girth 10?
    # If d=4, n = 2 * 5^4 = 1250.
    # The seed logic:
    # if (g-5) % 2 == 0: d = g - 4.
    # g=9 -> d=5.
    # This yields n=6250.
    # If I use d=4, n=1250.
    # Does CD(4, 5) have girth >= 9?
    # Girth is 2d+2 = 10. So yes.
    # So I should use d=4 for g=9.
    # The seed uses d=5, which is suboptimal.
    
    # Let's correct d to minimize n.
    # We need girth >= g.
    # Girth of CD(d, q) is 2d + 2.
    # So 2d + 2 >= g => d >= (g - 2) / 2.
    # d must be an integer.
    # d = ceil((g - 2) / 2).
    # For g=9, d = ceil(7/2) = 4.
    # For g=10, d = ceil(8/2) = 4.
    # For g=11, d = ceil(9/2) = 5.
    
    d = (g - 2 + 1) // 2  # ceil((g-2)/2)
    if d < 3:
        d = 3

    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add, sub, mul = (lambda x, y: x ^ y), (lambda x, y: x ^ y), (lambda x, y: table[x][y])
    else:
        add, sub, mul = (lambda x, y: (x + y) % q), (lambda x, y: (x - y) % q), (lambda x, y: x * y % q)
        
    # The rule for LUW graphs
    # rule[c] = (a, b) such that y[c] = y[a] * x[b] + x[c]
    # The seed uses a specific rule.
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
# EVOLVE-BLOCK-END
