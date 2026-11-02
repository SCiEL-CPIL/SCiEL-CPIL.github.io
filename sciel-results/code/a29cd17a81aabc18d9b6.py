"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
from collections import deque

# EVOLVE-BLOCK-START
def construct(k, g):
    deadline = time.time() + 110.0
    
    # Lazebnik-Ustimenko-Woldar graph CD(d, q) with q = k: an explicit q-regular bipartite graph
    # whose girth is at least d + 5 for odd d. Points and lines are vectors over GF(q) (q prime,
    # or q = 4); the connected component of the zero point is returned.
    q = k
    # For girth g, need d + 5 >= g, so d >= g - 5. d must be odd.
    if (g - 5) % 2 == 1:
        d = g - 5
    else:
        d = g - 4
    d = max(3, d)
    
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add = lambda x, y: x ^ y
        sub = lambda x, y: x ^ y
        mul = lambda x, y: table[x][y]
    else:
        add = lambda x, y: (x + y) % q
        sub = lambda x, y: (x - y) % q
        mul = lambda x, y: x * y % q
    
    # Precompute the rule for efficiency
    rule = {}
    rule[2] = (1, 1)
    rule[3] = (2, 1)
    rule[4] = (1, 2)
    for c in range(5, d + 1):
        r = (c - 5) % 4
        c0 = c - (c - 5) % 4
        options = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)]
        rule[c] = options[r]
    
    # Convert rules to lists for faster access
    rule_list = [None] * (d + 1)
    for c in range(2, d + 1):
        rule_list[c] = rule[c]
    
    # BFS to find connected component from zero point
    # Vertex representation: (side, tuple_of_coords) where side 0 = point, side 1 = line
    # For efficiency, use integer encoding where possible
    
    start = (0, tuple([None] + [0] * d))
    
    index = {start: 0}
    order = [start]
    edges = []
    
    # Precompute neighbor generation for speed
    for vertex in order:
        if time.time() > deadline:
            break
        side, x = vertex
        for first in range(q):
            # Build the neighbor coordinate by coordinate
            y_coords = [None, first] + [0] * (d - 1)
            for c in range(2, d + 1):
                a, b = rule_list[c]
                if side == 0:
                    # point p ~ line l: l[c] - p[c] = l[a] * p[b]
                    # so l[c] = p[c] + l[a] * p[b]
                    y_coords[c] = add(x[c], mul(y_coords[a], x[b]))
                else:
                    # line l ~ point p: l[c] - p[c] = l[a] * p[b]
                    # so p[c] = l[c] - l[a] * p[b]
                    y_coords[c] = sub(x[c], mul(x[a], y_coords[b]))
            other = (1 - side, tuple(y_coords))
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if side == 0:
                edges.append((index[vertex], index[other]))
    
    return len(order), edges
# EVOLVE-BLOCK-END
