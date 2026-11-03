import time

def construct(k, g):
    """
    Constructs a k-regular graph of girth at least g using the Lazebnik-Ustimenko-Woldar (LUW) construction.
    Returns (n, edges) where n is the number of vertices and edges is a list of (i, j) pairs.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin

    q = k

    # Determine d such that girth_of_d(d) >= g
    def girth_of_d(d):
        if d % 2 == 1:
            return d + 5
        else:
            return d + 6

    d = 3
    while girth_of_d(d) < g:
        d += 1

    # Precompute rules as tuples for speed
    rule_a = [0] * (d + 1)
    rule_b = [0] * (d + 1)
    rule_a[2], rule_b[2] = 1, 1
    rule_a[3], rule_b[3] = 2, 1
    rule_a[4], rule_b[4] = 1, 2
    for c in range(5, d + 1):
        r = (c - 5) % 4
        c0 = c - r
        if r == 0:
            rule_a[c], rule_b[c] = 1, c0 - 2
        elif r == 1:
            rule_a[c], rule_b[c] = c0 - 1, 1
        elif r == 2:
            rule_a[c], rule_b[c] = c0, 1
        else:
            rule_a[c], rule_b[c] = 1, c0 + 1
    
    # Convert to tuples for faster access
    rule_a_t = tuple(rule_a)
    rule_b_t = tuple(rule_b)

    # Generate vertices using BFS
    index = {}
    order = []
    edges = []

    # Start vertex: side 0, coordinates with x[0] as placeholder, x[1..d] = 0
    start_coord = (0,) * (d + 1)
    start_key = (0, start_coord)
    index[start_key] = 0
    order.append((0, start_coord))

    # Precompute range for c loop
    c_range = range(2, d + 1)

    for i, vertex in enumerate(order):
        side, x = vertex

        for first in range(q):
            # Build y as a list for mutation
            y = [0] * (d + 1)
            y[1] = first
            
            if side == 0:
                for c in c_range:
                    y[c] = (x[c] + y[rule_a_t[c]] * x[rule_b_t[c]]) % q
            else:
                for c in c_range:
                    y[c] = (x[c] - x[rule_a_t[c]] * y[rule_b_t[c]]) % q

            y_t = tuple(y)
            neighbor_key = (1 - side, y_t)

            if neighbor_key not in index:
                index[neighbor_key] = len(order)
                order.append((1 - side, y_t))

            if side == 0:
                edges.append((i, index[neighbor_key]))

        if time.time() > deadline:
            break

    n = len(order)
    return n, edges
