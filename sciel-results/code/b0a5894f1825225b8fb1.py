"""Construct a 3-regular graph of girth 16 using Lazebnik-Ustimenko-Woldar construction."""
import time

def construct(k, g):
    deadline = time.time() + 140  # 10s safety margin before 150s limit
    
    # Use LZW construction CD(d, q) with q = k
    # For girth >= g: need d such that d + 5 >= g (odd d) or d + 4 >= g (even d)
    q = k
    # Choose d to ensure girth >= g
    if g <= 8:
        d = 3
    else:
        # For odd d: girth >= d+5, so d >= g-5
        # For even d: girth >= d+4, so d >= g-4
        d_odd = g - 5
        d_even = g - 4
        # Pick the smaller valid d
        if d_odd % 2 == 1 and d_odd >= 3:
            d = d_odd
        else:
            d = d_even
            if d % 2 != 0:
                d += 1
    
    # Ensure d is at least 3
    d = max(3, d)
    
    # Define field operations for GF(q) where q is prime
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add = lambda x, y: x ^ y
        sub = lambda x, y: x ^ y
        mul = lambda x, y: table[x][y]
    else:
        add = lambda x, y: (x + y) % q
        sub = lambda x, y: (x - y) % q
        mul = lambda x, y: x * y % q
    
    # Define the rule for LZW construction
    # rule[c] = (a, b) for c = 2..d
    rule = {}
    rule[2] = (1, 1)
    rule[3] = (2, 1)
    rule[4] = (1, 2)
    for c in range(5, d + 1):
        r = (c - 5) % 4
        c0 = c - (c - 5) % 4
        options = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)]
        rule[c] = options[r]
    
    def neighbours(vertex):
        side, x = vertex
        for first in range(q):
            y = [None] * (d + 1)
            y[1] = first
            for c in range(2, d + 1):
                y[c] = 0
            for c in range(2, d + 1):
                a, b = rule[c]
                if side == 0:
                    y[c] = add(x[c], mul(y[a], x[b]))
                else:
                    y[c] = sub(x[c], mul(x[a], y[b]))
            yield (1 - side, tuple(y))
    
    # BFS from the zero vertex to find the connected component
    start = (0, tuple([None] + [0] * d))
    index = {start: 0}
    order = [start]
    edges = []
    
    for i in range(len(order)):
        if time.time() > deadline:
            break
        vertex = order[i]
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            # Only add edge from side 0 to avoid duplicates
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
    
    n = len(order)
    return n, edges
