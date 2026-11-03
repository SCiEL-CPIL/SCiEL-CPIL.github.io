import time
import sys

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110  # 10s margin
    
    # Lazebnik-Ustimenko-Woldar graph CD(d, q)
    # q = k = 5
    q = k
    # d is chosen such that girth >= g
    # Girth of CD(d, q) is d + 5 for odd d?
    # If g=9, we need girth >= 9.
    # If d=4, girth=9.
    # Seed logic: d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    # For g=9: g-5=4 (even) -> d = 9-4=5.
    # If d=5, girth = 5+5=10.
    # So seed produces girth 10 graph.
    
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
    # The seed computes rule[c] for c in 2..d
    rule = {}
    rule[2] = (1, 1)
    rule[3] = (2, 1)
    rule[4] = (1, 2)
    for c in range(5, d + 1):
        r, c0 = (c - 5) % 4, c - (c - 5) % 4
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]

    # Pre-compute neighbors for a given vertex?
    # Vertex is (side, x) where x is a tuple of length d+1 (indices 0..d)
    # side 0: Point. side 1: Line.
    # x[0] is None for points?
    # Seed: start = (0, (None,) + (0,) * d)
    # So x has length d+1.
    
    # To speed up, we can use a dictionary for visited vertices.
    # The number of vertices is roughly q^d?
    # For q=5, d=5, 5^5 = 3125.
    # But the graph is a component, so it might be smaller or larger?
    # The seed says "connected component of the zero point".
    
    index = {}
    order = []
    edges = []
    
    start = (0, (None,) + (0,) * d)
    index[start] = 0
    order.append(start)
    
    # BFS
    for i in range(len(order)):
        if time.time() > deadline:
            break
        vertex = order[i]
        side, x = vertex
        for first in range(q):
            # Construct y
            # y is a tuple of length d+1
            y_list = [None, first] + [0] * (d - 1)
            for c in range(2, d + 1):
                a, b = rule[c]
                if side == 0:
                    # Point to Line
                    # l[c] - p[c] = l[a] * p[b]
                    # y[c] = x[c] + y[a] * x[b]
                    y_list[c] = add(x[c], mul(y_list[a], x[b]))
                else:
                    # Line to Point
                    # p[c] = l[c] - l[a] * p[b]? 
                    # Seed: y[c] = sub(x[c], mul(x[a], y[b]))
                    # Here x is the line, y is the point being constructed?
                    # No, 'other' is the neighbor.
                    # If side==1 (Line), we are finding Points.
                    # y is the point coordinates.
                    # x is the line coordinates.
                    # Relation: x[c] - y[c] = x[a] * y[b]?
                    # Seed: y[c] = x[c] - x[a] * y[b]
                    y_list[c] = sub(x[c], mul(x[a], y_list[b]))
            
            other = (1 - side, tuple(y_list))
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if side == 0:
                edges.append((index[vertex], index[other]))
                
    return len(order), edges
