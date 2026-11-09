"""Construct 3-regular graph of girth 16 using Lazebnik-Ustimenko-Woldar construction."""
import time
import sys

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    # Lazebnik-Ustimenko-Woldar graph CD(d, q)
    # For k=3, g=16: use q=3, d=11 (girth >= d+5 = 16)
    q = k
    # d must be odd and >= g-5
    if (g - 5) % 2 == 0:
        d = g - 4
    else:
        d = g - 5
    d = max(d, 3)
    
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add = lambda x, y: x ^ y
        sub = lambda x, y: x ^ y
        mul = lambda x, y: table[x][y]
    else:
        add = lambda x, y: (x + y) % q
        sub = lambda x, y: (x - y) % q
        mul = lambda x, y: x * y % q
    
    # Build the rule: rule[c] = (a, b) meaning y[c] depends on y[a] and x[b]
    # For LUW: rule[2]=(1,1), rule[3]=(2,1), rule[4]=(1,2)
    # For c >= 5: pattern based on (c-5) mod 4
    rule = {}
    rule[2] = (1, 1)
    rule[3] = (2, 1)
    rule[4] = (1, 2)
    for c in range(5, d + 1):
        r = (c - 5) % 4
        c0 = c - r
        # The four cases for r=0,1,2,3
        if r == 0:
            rule[c] = (1, c0 - 2)
        elif r == 1:
            rule[c] = (c0 - 1, 1)
        elif r == 2:
            rule[c] = (c0, 1)
        else:
            rule[c] = (1, c0 + 1)
    
    # Precompute rule arrays for fast access
    rule_a = [0] * (d + 1)
    rule_b = [0] * (d + 1)
    for c in range(2, d + 1):
        rule_a[c], rule_b[c] = rule[c]
    
    # Generate neighbours using precomputed rules
    def get_neighbours(side, x):
        result = []
        for first in range(q):
            y = [0] * (d + 1)
            y[1] = first
            if side == 0:
                for c in range(2, d + 1):
                    y[c] = add(x[c], mul(y[rule_a[c]], x[rule_b[c]]))
            else:
                for c in range(2, d + 1):
                    y[c] = sub(x[c], mul(x[rule_a[c]], y[rule_b[c]]))
            result.append((1 - side, tuple(y)))
        return result
    
    # BFS from start vertex to get connected component
    start_side = 0
    start_x = tuple([0] * (d + 1))
    start = (start_side, start_x)
    index = {start: 0}
    order = [start]
    edges = []
    
    for i, vertex in enumerate(order):
        if time.time() > deadline:
            break
        side, x = vertex
        for other in get_neighbours(side, x):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if side == 0:
                edges.append((i, index[other]))
    
    n = len(order)
    return n, edges
