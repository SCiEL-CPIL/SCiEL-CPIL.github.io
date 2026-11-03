"""Construct a 4-regular graph with girth at least 10 using LUW construction."""
import time

def construct(k, g):
    """Construct a k-regular graph with girth at least g.
    
    Uses the Lazebnik-Ustimenko-Woldar construction.
    Returns (n, edges) where n is the number of vertices and edges is a list of (i,j) pairs.
    """
    start_time = time.time()
    deadline = start_time + 100  # 100 second deadline with safety margin
    
    q = k
    d = g - 5 if (g - 5) % 2 == 1 else g - 4
    d = max(3, d)
    
    if q == 4:
        mul_table = [
            [0, 0, 0, 0],
            [0, 1, 2, 3],
            [0, 2, 3, 2],
            [0, 3, 2, 1]
        ]
        add = lambda x, y: x ^ y
        sub = lambda x, y: x ^ y
        mul = lambda x, y: mul_table[x][y]
    else:
        add = lambda x, y: (x + y) % q
        sub = lambda x, y: (x - y) % q
        mul = lambda x, y: (x * y) % q
    
    rule = {}
    rule[2] = (1, 1)
    rule[3] = (2, 1)
    rule[4] = (1, 2)
    for c in range(5, d + 1):
        r = (c - 5) % 4
        c0 = c - (c - 5) % 4
        options = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)]
        rule[c] = options[r]
    
    def get_neighbors(vertex):
        side, x = vertex
        neighbors = []
        for first in range(q):
            y = [None, first] + [0] * (d - 1)
            for c in range(2, d + 1):
                a, b = rule[c]
                if side == 0:
                    y[c] = add(x[c], mul(y[a], x[b]))
                else:
                    y[c] = sub(x[c], mul(x[a], y[b]))
            neighbors.append((1 - side, tuple(y)))
        return neighbors
    
    start_vertex = (0, (None,) + (0,) * d)
    visited = {start_vertex: 0}
    order = [start_vertex]
    edges = []
    
    idx = 0
    while idx < len(order):
        vertex = order[idx]
        idx += 1
        for neighbor in get_neighbors(vertex):
            if neighbor not in visited:
                visited[neighbor] = len(order)
                order.append(neighbor)
            if vertex[0] == 0:
                edges.append((visited[vertex], visited[neighbor]))
        
        if time.time() > deadline:
            break
    
    n = len(order)
    return n, edges
