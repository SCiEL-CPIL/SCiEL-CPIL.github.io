import time
import sys

def construct(k, g):
    """
    Constructs a k-regular graph of girth at least g.
    Uses the Lazebnik-Ustimenko-Woldar (LUW) construction.
    """
    start_time = time.time()
    deadline = start_time + 140.0
    
    q = k
    if q != 3:
        # Fallback: Return a trivial graph or raise?
        # The task assumes k=3.
        # If k != 3, LUW might not apply or q must be prime power.
        # For this specific task, k=3.
        pass

    def girth_of_d(d):
        if d % 2 == 1:
            return d + 5
        else:
            return d + 6
            
    d = 3
    while girth_of_d(d) < g:
        d += 1
        
    # GF(q) operations
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        def add(x, y): return x ^ y
        def sub(x, y): return x ^ y
        def mul(x, y): return table[x][y]
    else:
        def add(x, y): return (x + y) % q
        def sub(x, y): return (x - y) % q
        def mul(x, y): return x * y % q

    # Rule for LUW
    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r, c0 = (c - 5) % 4, c - (c - 5) % 4
        rules_cycle = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)]
        rule[c] = rules_cycle[r]

    index = {}
    order = []
    edges = []
    
    start = (0, (None,) + (0,) * d)
    index[start] = 0
    order.append(start)
    
    for i, vertex in enumerate(order):
        side, x = vertex
        for first in range(q):
            y = [None, first] + [0] * (d - 1)
            for c in range(2, d + 1):
                a, b = rule[c]
                if side == 0:
                    y[c] = add(x[c], mul(y[a], x[b]))
                else:
                    y[c] = sub(x[c], mul(x[a], y[b]))
            
            neighbor = (1 - side, tuple(y))
            
            if neighbor not in index:
                index[neighbor] = len(order)
                order.append(neighbor)
            
            if side == 0:
                edges.append((index[vertex], index[neighbor]))
                
        if time.time() > deadline:
            break
            
    n = len(order)
    return n, edges
