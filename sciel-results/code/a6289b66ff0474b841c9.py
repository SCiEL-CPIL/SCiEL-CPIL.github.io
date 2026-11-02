"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct(k, g):
    # Lazebnik-Ustimenko-Woldar graph CD(d, q) with q = k.
    # This construction is known to produce cages for certain parameters.
    # For k=4, g=10, we use q=4 and d=5.
    q = k
    # Determine d such that girth is at least g.
    # Girth of CD(d, q) is at least d + 5.
    # We need d + 5 >= g => d >= g - 5.
    # Also d must be odd for the specific LUW variant used here?
    # The seed logic: d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    # For g=10, g-5=5 (odd), so d=5.
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    
    # Define GF(q) operations
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add = lambda x, y: x ^ y
        sub = lambda x, y: x ^ y
        mul = lambda x, y: table[x][y]
    else:
        add = lambda x, y: (x + y) % q
        sub = lambda x, y: (x - y) % q
        mul = lambda x, y: x * y % q

    # Define the rule for the LUW graph
    # This part is copied from the seed and assumed correct for the family.
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
                # Point p ~ Line l iff l[c] - p[c] = l[a] * p[b]
                if side == 0:
                    y[c] = add(x[c], mul(y[a], x[b]))
                else:
                    y[c] = sub(x[c], mul(x[a], y[b]))
            yield 1 - side, tuple(y)

    # BFS to find the connected component of the zero vertex
    start = (0, (None,) + (0,) * d)
    index = {start: 0}
    order = [start]
    edges = []
    
    for i, vertex in enumerate(order):
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            # Add edge if we are on the point side (side 0) to avoid duplicates
            if vertex[0] == 0:
                edges.append((i, index[other]))
                
    return len(order), edges
# EVOLVE-BLOCK-END
