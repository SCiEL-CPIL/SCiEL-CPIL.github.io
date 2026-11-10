"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct(k, g):
    # Lazebnik-Ustimenko-Woldar graph CD(d, q) with q = k: an explicit q-regular bipartite graph
    # whose girth is at least d + 5 for odd d. Points and lines are vectors over GF(q) (q prime,
    # or q = 4); the connected component of the zero point is returned. Valid, far from optimal.
    deadline = time.time() + 140  # Safety margin before 150s limit
    
    q = k
    # For girth g, need d+5 >= g with d odd. Minimal d:
    if (g - 5) % 2 == 1:
        d = max(3, g - 5)
    else:
        d = max(3, g - 4)
    
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add, sub, mul = (lambda x, y: x ^ y), (lambda x, y: x ^ y), (lambda x, y: table[x][y])
    else:
        add, sub, mul = (lambda x, y: (x + y) % q), (lambda x, y: (x - y) % q), (lambda x, y: x * y % q)
    
    # Precompute rule as a list for faster access (index 0 and 1 unused)
    rule = [None, None, (1, 1), (2, 1), (1, 2)]
    for c in range(5, d + 1):
        r, c0 = (c - 5) % 4, c - (c - 5) % 4
        rule.append([(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r])
    
    start = (0, (None,) + (0,) * d)
    index = {start: 0}
    order = [start]
    edges = []
    
    for i, vertex in enumerate(order):
        if time.time() > deadline:
            break
        side, x = vertex
        for first in range(q):
            y_list = [None, first] + [0] * (d - 1)
            for c in range(2, d + 1):
                a, b = rule[c]
                if side == 0:
                    y_list[c] = add(x[c], mul(y_list[a], x[b]))
                else:
                    y_list[c] = sub(x[c], mul(x[a], y_list[b]))
            other = (1 - side, tuple(y_list))
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if side == 0:
                edges.append((index[vertex], index[other]))
    
    return len(order), edges
# EVOLVE-BLOCK-END
