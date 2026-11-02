"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def construct(k, g):
    """
    Construct a k-regular graph with girth at least g.
    Uses Lazebnik-Ustimenko-Woldar (LUW) construction CD(d, q).
    """
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # LUW parameters
    # For odd girth g, we need d such that d + 5 >= g.
    # Smallest d is g - 5.
    # If g - 5 is even, we might be able to use a smaller d?
    # The girth of CD(d, q) is exactly d + 5 if d is odd? Or at least?
    # The paper states girth is at least d + 5.
    # To minimize vertices n = 2 * q^(d+1), we want smallest d.
    d = g - 5
    if d < 1:
        d = 1
        
    q = k
    
    # Field operations
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add = lambda x, y: x ^ y
        sub = lambda x, y: x ^ y
        mul = lambda x, y: table[x][y]
    else:
        add = lambda x, y: (x + y) % q
        sub = lambda x, y: (x - y) % q
        mul = lambda x, y: x * y % q

    # Rule for LUW
    # rule[c] = (a, b) for c = 2..d
    # The recurrence is y[c] = y[c-1] + x[a] * y[b] (simplified for explanation)
    # Actually, the standard LUW rule for c=2..d is:
    # l[c] = p[c] + l[a] * p[b]
    # with (a,b) depending on c.
    
    # From Lazebnik et al. (1996):
    # For c=2, (a,b) = (1,1) ? No, indices are 1..d.
    # Let's use the specific rule from the seed code which is known to work.
    rule = {}
    # The seed code used:
    # rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    # for c in 5..d:
    #   r, c0 = (c-5)%4, c - (c-5)%4
    #   rule[c] = [(1, c0-2), (c0-1, 1), (c0, 1), (1, c0+1)][r]
    
    # I will replicate this logic exactly to ensure validity.
    if d >= 2: rule[2] = (1, 1)
    if d >= 3: rule[3] = (2, 1)
    if d >= 4: rule[4] = (1, 2)
    for c in range(5, d + 1):
        r = (c - 5) % 4
        c0 = c - r
        options = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)]
        rule[c] = options[r]

    def neighbours(vertex):
        side, x = vertex
        # x is a tuple of length d+1, x[0] is unused (or 0), x[1]..x[d] are coords
        # In seed code: x = (None,) + (0,)*d
        # So x[1] is first coord, x[d] is last.
        
        for first in range(q):
            # y[1] = first
            # y[c] for c=2..d
            y = [None] * (d + 1)
            y[1] = first
            # y[0] is ignored/None
            
            for c in range(2, d + 1):
                a, b = rule[c]
                if side == 0:
                    # Point to Line
                    # y[c] = x[c] + y[a] * x[b]
                    y[c] = add(x[c], mul(y[a], x[b]))
                else:
                    # Line to Point
                    # y[c] = x[c] - x[a] * y[b]
                    y[c] = sub(x[c], mul(x[a], y[b]))
            
            yield 1 - side, tuple(y)

    # BFS to get connected component
    start = (0, tuple([None] + [0] * d))
    index = {start: 0}
    order = [start]
    edges = []
    
    i = 0
    while i < len(order):
        if time.time() > deadline:
            break
        vertex = order[i]
        i += 1
        v_idx = index[vertex]
        
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            
            # Add edge only if from point to line to avoid duplicates
            if vertex[0] == 0:
                edges.append((v_idx, index[other]))

    n = len(order)
    return n, edges

# EVOLVE-BLOCK-END
