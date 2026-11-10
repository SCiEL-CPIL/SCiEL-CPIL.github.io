"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # Lazebnik-Ustimenko-Woldar graph CD(d, q)
    q = k
    # d is chosen such that girth >= d + 5
    # For girth g, we need d >= g - 5
    # If g - 5 is odd, use g - 5. If even, use g - 4?
    # The LUW graph CD(d, q) has girth at least d + 5 for odd d.
    # For even d, girth is at least d + 4?
    # Let's use d = g - 5 if (g - 5) % 2 == 1 else g - 4
    d = g - 5
    if d % 2 == 0:
        d = d + 1 # Make it odd? Or just use d?
        # Actually, for g=16, g-5=11 (odd). So d=11.
        
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add = lambda x, y: x ^ y
        sub = lambda x, y: x ^ y
        mul = lambda x, y: table[x][y]
    else:
        add = lambda x, y: (x + y) % q
        sub = lambda x, y: (x - y) % q
        mul = lambda x, y: x * y % q

    rule = {}
    rule[2] = (1, 1)
    rule[3] = (2, 1)
    rule[4] = (1, 2)
    for c in range(5, d + 1):
        r, c0 = (c - 5) % 4, c - (c - 5) % 4
        # The rule indices in the original code:
        # [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]
        if r == 0:
            rule[c] = (1, c0 - 2)
        elif r == 1:
            rule[c] = (c0 - 1, 1)
        elif r == 2:
            rule[c] = (c0, 1)
        else:
            rule[c] = (1, c0 + 1)

    # Precompute neighbor offsets?
    # Vertex is (side, x_1, ..., x_d)
    # We represent x as a tuple of integers.
    
    # To speed up, we can use a dictionary for visited nodes.
    # Start vertex: side 0, all zeros.
    start_vertex = (0,) + (0,) * d
    index = {start_vertex: 0}
    order = [start_vertex]
    edges = []
    
    # BFS
    for i, vertex in enumerate(order):
        if time.time() > deadline:
            break
        side, x = vertex[0], vertex[1:]
        # Generate neighbors
        for first in range(q):
            # y is the neighbor's coordinates
            # y[1] = first
            # y[0] is None (side indicator handled separately)
            y_coords = [0] * (d + 1)
            y_coords[1] = first
            # y[2..d]
            for c in range(2, d + 1):
                a, b = rule[c]
                if side == 0:
                    # point p ~ line l
                    # l[c] - p[c] = l[a] * p[b]
                    # y[c] = x[c] + y[a] * x[b]
                    # Note: indices in x are 1-based in tuple, 0-based in list?
                    # x is tuple of length d. x[0] is x_1.
                    # rule[c] gives a, b in 1..d
                    # So x[a-1], x[b-1]
                    # y[a-1], y[b-1]
                    ya = y_coords[a]
                    xb = x[b-1]
                    y_coords[c] = add(x[c-1], mul(ya, xb))
                else:
                    # line l ~ point p
                    # l[c] - p[c] = l[a] * p[b]
                    # x[c] - y[c] = x[a] * y[b]
                    # y[c] = x[c] - x[a] * y[b]
                    ya = y_coords[a]
                    xa = x[a-1]
                    y_coords[c] = sub(x[c-1], mul(xa, ya))
            
            neighbor = (1 - side, tuple(y_coords[1:]))
            
            if neighbor not in index:
                index[neighbor] = len(order)
                order.append(neighbor)
            
            if side == 0:
                edges.append((index[vertex], index[neighbor]))
                
    return len(order), edges
# EVOLVE-BLOCK-END
