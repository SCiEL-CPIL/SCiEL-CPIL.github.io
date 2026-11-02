import time
import sys

def construct(k, g):
    start_time = time.time()
    deadline = 145.0 # Safety margin
    
    # Lazebnik-Ustimenko-Woldar graph CD(d, q) with q = k
    q = k
    # For girth g, we need d such that d+5 >= g (for odd d) or d+4 >= g (for even d?)
    # The seed logic: d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    # For g=10, g-5=5 (odd), so d=5.
    # Girth of CD(d,q) is d+5 for odd d? 
    # Actually, the girth of CD(d,q) is 2d+2 if d is even? No.
    # Standard result: Girth of CD(d, q) is d+5 if d is odd, and d+4 if d is even?
    # Let's stick to the seed's formula which was likely derived from the paper.
    
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add = lambda x, y: x ^ y
        sub = lambda x, y: x ^ y
        mul = lambda x, y: table[x][y]
    else:
        add = lambda x, y: (x + y) % q
        sub = lambda x, y: (x - y) % q
        mul = lambda x, y: x * y % q
        
    # Rule for LUW graph
    # The rule defines the connection between points and lines.
    # l[c] - p[c] = l[a] * p[b]
    rule = {}
    # The seed has a specific rule generation.
    # rule[2] = (1, 1)
    # rule[3] = (2, 1)
    # rule[4] = (1, 2)
    # for c in 5..d:
    #   r = (c-5)%4
    #   c0 = c - r
    #   rule[c] = [(1, c0-2), (c0-1, 1), (c0, 1), (1, c0+1)][r]
    
    rule[2] = (1, 1)
    rule[3] = (2, 1)
    rule[4] = (1, 2)
    for c in range(5, d + 1):
        r = (c - 5) % 4
        c0 = c - r
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]

    def neighbours(vertex):
        side, x = vertex
        # x is a tuple of length d+1 (index 0 is None/side marker?)
        # In seed: start = (0, (None,) + (0,) * d)
        # So x[0] is None, x[1..d] are coordinates.
        # Wait, the seed uses x[c] for c in 2..d.
        # So x[0] is None, x[1] is first coord?
        # Let's check the seed's neighbour generation.
        # y = [None, first] + [0] * (d - 1)
        # y has length 1 + 1 + d - 1 = d + 1.
        # Indices 0..d.
        # y[0] = None.
        # y[1] = first (variable).
        # y[2..d] = 0 initially.
        
        for first in range(q):
            y = [None, first] + [0] * (d - 1)
            for c in range(2, d + 1):
                a, b = rule[c]
                if side == 0:
                    # Point p=x, Line l=y.
                    # l[c] - p[c] = l[a] * p[b]
                    # y[c] = x[c] + y[a] * x[b]
                    y[c] = add(x[c], mul(y[a], x[b]))
                else:
                    # Line l=x, Point p=y.
                    # l[c] - p[c] = l[a] * p[b]
                    # x[c] - y[c] = x[a] * y[b]
                    # y[c] = x[c] - x[a] * y[b]
                    y[c] = sub(x[c], mul(x[a], y[b]))
            yield 1 - side, tuple(y)

    start = (0, (None,) + (0,) * d)
    index = {start: 0}
    order = [start]
    edges = []
    
    # BFS to find connected component
    for i, vertex in enumerate(order):
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
                
    # Check time
    if time.time() > deadline:
        # Return best effort
        pass
        
    return len(order), edges
