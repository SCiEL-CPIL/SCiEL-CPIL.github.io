"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct(k, g):
    # Lazebnik-Ustimenko-Woldar graph CD(d, q) with q = k.
    # This is a q-regular bipartite graph with girth 2d + 2.
    # We choose the smallest d such that 2d + 2 >= g.
    q = k
    # d must be an integer >= 1
    d = (g - 2 + 1) // 2  # ceil((g-2)/2)
    if d < 1:
        d = 1
        
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add, sub, mul = (lambda x, y: x ^ y), (lambda x, y: x ^ y), (lambda x, y: table[x][y])
    else:
        # q is prime (k=5)
        add, sub, mul = (lambda x, y: (x + y) % q), (lambda x, y: (x - y) % q), (lambda x, y: x * y % q)
        
    # Define the rule for LUW graph CD(d, q)
    # The rule maps index c (2..d) to (a, b) such that l[c] - p[c] = l[a] * p[b]
    # Standard LUW rule:
    # For c = 2..d, the coefficients are defined by the "cyclic" structure.
    # In the original paper, the rule is:
    # l_i = p_i + p_{i-1} * p_{i-2} ... ? No, it's a specific linear recurrence over the field.
    # The code provided in the prompt uses a specific rule dictionary.
    # I will reuse the rule generation from the prompt to ensure correctness.
    
    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r, c0 = (c - 5) % 4, c - (c - 5) % 4
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]

    def neighbours(vertex):
        side, x = vertex
        for first in range(q):
            # x is a tuple of length d+1: x[0] is None, x[1]..x[d] are values
            y = [None, first] + [0] * (d - 1)
            for c in range(2, d + 1):
                a, b = rule[c]
                # point p ~ line l  iff  l[c] - p[c] = l[a] * p[b]
                # If side == 0 (point), x is point, y is line.
                # y[c] = x[c] + y[a] * x[b]
                # If side == 1 (line), x is line, y is point.
                # y[c] = x[c] - x[a] * y[b]
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
