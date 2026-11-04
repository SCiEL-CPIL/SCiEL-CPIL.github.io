"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct(k, g):
    deadline = time.time() + 120  # Safety margin before 150s limit
    q = k
    
    # Choose smallest d achieving girth >= g
    # L-U-W CD(d,q): girth >= d+5 if d odd, d+6 if d even
    if g % 2 == 0:
        # Even g: d = g-5 works (girth >= g if d odd, >= g+1 if d even)
        d = g - 5
    else:
        # Odd g: need d odd with d+5 >= g, so d >= g-5
        d = g - 5
        if d % 2 == 0:
            d += 1
    d = max(3, d)
    
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add, sub, mul = (lambda x, y: x ^ y), (lambda x, y: x ^ y), (lambda x, y: table[x][y])
    else:
        add, sub, mul = (lambda x, y: (x + y) % q), (lambda x, y: (x - y) % q), (lambda x, y: x * y % q)
    
    # Precompute rule table for efficiency
    rule = {}
    rule[2] = (1, 1)
    rule[3] = (2, 1)
    rule[4] = (1, 2)
    for c in range(5, d + 1):
        r = (c - 5) % 4
        c0 = c - r
        if r == 0:
            rule[c] = (1, c0 - 2)
        elif r == 1:
            rule[c] = (c0 - 1, 1)
        elif r == 2:
            rule[c] = (c0, 1)
        else:
            rule[c] = (1, c0 + 1)
    
    # Precompute rules as a list indexed by c for faster access
    rules = [None] * (d + 1)
    for c in range(2, d + 1):
        rules[c] = rule[c]
    
    def neighbours(vertex):
        side, x = vertex
        x0, x1 = x[0], x[1]
        for first in range(q):
            # Build y array incrementally
            y = [0] * (d + 1)
            y[0] = None
            y[1] = first
            # y[2..d] computed below
            
            if side == 0:
                # Point side: y[c] = x[c] + y[a] * x[b]
                for c in range(2, d + 1):
                    a, b = rules[c]
                    y[c] = (x[c] + y[a] * x[b]) % q if q != 4 else add(x[c], mul(y[a], x[b]))
            else:
                # Line side: y[c] = x[c] - x[a] * y[b]
                for c in range(2, d + 1):
                    a, b = rules[c]
                    y[c] = (x[c] - x[a] * y[b]) % q if q != 4 else sub(x[c], mul(x[a], y[b]))
            
            yield 1 - side, tuple(y)
    
    # BFS to find connected component
    start = (0, (None,) + (0,) * d)
    index = {start: 0}
    order = [start]
    edges = []
    
    for i, vertex in enumerate(order):
        if time.time() > deadline:
            break
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
    
    n = len(order)
    return n, edges
# EVOLVE-BLOCK-END
