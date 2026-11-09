import time
import sys

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # LUW graph parameters
    q = k
    # For bipartite graphs, girth is even. If g is odd, we target girth >= g+1 (even)
    # The LUW graph CD(d, q) has girth 2d+2? Or related to d.
    # Seed code: d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    # Let's stick to the seed's d calculation as it's a known mapping.
    if (g - 5) % 2:
        d = g - 5
    else:
        d = g - 4
    if d < 3:
        d = 3
        
    # Finite field GF(q) operations
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        def add(x, y): return x ^ y
        def sub(x, y): return x ^ y
        def mul(x, y): return table[x][y]
    else:
        def add(x, y): return (x + y) % q
        def sub(x, y): return (x - y) % q
        def mul(x, y): return (x * y) % q

    # Rule generation for LUW
    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r = (c - 5) % 4
        c0 = c - (c - 5) % 4
        # The seed used a list index: [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]
        # Let's replicate this exactly to ensure correctness.
        options = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)]
        rule[c] = options[r]

    # Neighbor generation
    def neighbours(vertex):
        side, x = vertex
        # x is a tuple of length d+1, x[0] is None, x[1..d] are values in GF(q)
        for first in range(q):
            # y is a tuple for the other side
            # y[0] is None, y[1] = first, y[2..d] are computed
            y_list = [None, first] + [0] * (d - 1)
            for c in range(2, d + 1):
                a, b = rule[c]
                if side == 0:
                    # Point to Line: l[c] - p[c] = l[a] * p[b] => l[c] = p[c] + l[a]*p[b]
                    # Here x is Point, y is Line.
                    # y[c] = x[c] + y[a] * x[b]
                    y_list[c] = add(x[c], mul(y_list[a], x[b]))
                else:
                    # Line to Point: p[c] = l[c] - l[a]*p[b] ? 
                    # Seed: sub(x[c], mul(x[a], y[b]))
                    # x is Line, y is Point.
                    y_list[c] = sub(x[c], mul(x[a], y_list[b]))
            yield 1 - side, tuple(y_list)

    # BFS to get connected component
    start = (0, (None,) + (0,) * d)
    index = {start: 0}
    order = [start]
    edges = []
    
    i = 0
    while i < len(order):
        if time.time() > deadline:
            break
        vertex = order[i]
        i += 1
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
                
    n = len(order)
    return n, edges
