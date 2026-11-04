"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
# EVOLVE-BLOCK-START
def construct(k, g):
    """Lazebnik-Ustimenko-Woldar graph CD(d, q) with q = k.
    
    Returns the connected component of the zero point, which for odd d >= 3
    has girth at least d+5. We choose d so that d+5 >= g, minimizing d.
    The component size is 2*(1 + q + q^2 + ... + q^((d-1)/2)) when the
    girth is exactly d+5 (i.e., the Moore bound).
    """
    deadline = time.time() + 140  # safety margin before 150s kill
    
    q = k
    # For odd d, girth >= d+5. We need d+5 >= g, so d >= g-5.
    # d must be odd for the standard LUW construction.
    d = g - 5
    if d < 3:
        d = 3
    if d % 2 == 0:
        d += 1  # ensure odd
    
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add, sub, mul = (lambda x, y: x ^ y), (lambda x, y: x ^ y), (lambda x, y: table[x][y])
    else:
        add, sub, mul = (lambda x, y: (x + y) % q), (lambda x, y: (x - y) % q), (lambda x, y: x * y % q)
    
    # Precompute rule table
    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r, c0 = (c - 5) % 4, c - (c - 5) % 4
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]
    
    # Precompute rule arrays for speed
    rule_a = [0] * (d + 1)
    rule_b = [0] * (d + 1)
    for c in range(2, d + 1):
        rule_a[c], rule_b[c] = rule[c]
    
    def neighbours_fast(vertex):
        side, x = vertex
        for first in range(q):
            y = [None, first] + [0] * (d - 1)
            if side == 0:
                # point to line: y[c] = x[c] + y[a] * x[b]
                for c in range(2, d + 1):
                    a, b = rule_a[c], rule_b[c]
                    y[c] = (x[c] + y[a] * x[b]) % q
            else:
                # line to point: y[c] = x[c] - x[a] * y[b]
                for c in range(2, d + 1):
                    a, b = rule_a[c], rule_b[c]
                    y[c] = (x[c] - x[a] * y[b]) % q
            yield 1 - side, tuple(y)
    
    start = (0, (None,) + (0,) * d)
    index = {start: 0}
    order = [start]
    edges = []
    
    # BFS to find connected component
    i = 0
    while i < len(order):
        vertex = order[i]
        i += 1
        for other in neighbours_fast(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
        
        if time.time() > deadline:
            break
    
    n = len(order)
    return n, edges
# EVOLVE-BLOCK-END
