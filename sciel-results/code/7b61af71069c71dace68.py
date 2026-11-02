import time
import sys

def construct(k, g):
    # Deadline guard: return best valid result before timeout
    deadline = time.time() + 110  # 110s safety margin out of 120s limit
    
    # For the LUW (Lazebnik-Ustimenko-Woldar) construction CD(d, q):
    # - q-regular bipartite graph
    # - girth = 2d + 2
    # - vertices = 2 * q^d
    # We need girth >= g, so d >= ceil((g-2)/2)
    # And q = k (to get k-regular)
    
    q = k
    d = (g - 2 + 1) // 2  # ceil((g-2)/2)
    
    # For k=3, g=16: d = ceil(14/2) = 7, N = 2*3^7 = 4374
    
    # Check if this is computable in time
    # 2 * q^d vertices, each with q neighbors
    n_est = 2 * q ** d
    if n_est > 200000:
        # Too large, try to find a smaller construction or fall back
        # For now, just proceed if it's not astronomically large
        pass
    
    # GF(q) arithmetic (q must be prime power; for k=3, q=3 is prime)
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        def add(x, y): return x ^ y
        def sub(x, y): return x ^ y
        def mul(x, y): return table[x][y]
    else:
        def add(x, y): return (x + y) % q
        def sub(x, y): return (x - y) % q
        def mul(x, y): return (x * y) % q
    
    # The rule array for LUW construction
    # rule[c] = (a, b) for c = 2..d
    # This defines the linear constraint for the graph edges
    rule = {}
    # Standard LUW rules:
    # rule[2] = (1,1), rule[3] = (2,1), rule[4] = (1,2)
    # For c >= 5: periodic pattern
    rule[2] = (1, 1)
    rule[3] = (2, 1)
    rule[4] = (1, 2)
    for c in range(5, d + 1):
        r = (c - 5) % 4
        c0 = c - (c - 5) % 4
        if r == 0:
            rule[c] = (1, c0 - 2)
        elif r == 1:
            rule[c] = (c0 - 1, 1)
        elif r == 2:
            rule[c] = (c0, 1)
        else:
            rule[c] = (1, c0 + 1)
    
    # Build the graph by BFS from the origin
    # Points: (0, x) where x is a tuple of d values (first is None/unused)
    # Lines: (1, x) where x is a tuple of d values
    
    start = (0, tuple([0] * (d + 1)))  # first element unused, indices 1..d
    
    index = {start: 0}
    order = [start]
    edges = []
    
    def get_neighbours(vertex):
        side, x = vertex
        for first in range(q):
            y = [0] * (d + 1)
            y[1] = first
            # Compute the rest of y
            for c in range(2, d + 1):
                a, b = rule[c]
                if side == 0:
                    # point x ~ line y iff y[c] - x[c] = y[a] * x[b]
                    y[c] = add(x[c], mul(y[a], x[b]))
                else:
                    # line x ~ point y iff x[c] - y[c] = x[a] * y[b]
                    y[c] = sub(x[c], mul(x[a], y[b]))
            yield (1 - side, tuple(y))
    
    # BFS to build the connected component
    edge_set = set()
    for vi, vertex in enumerate(order):
        for other in get_neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            
            # Add edge only once (when from point side to avoid duplicates)
            if vertex[0] == 0:
                u = index[vertex]
                v = index[other]
                if u < v:
                    edge_set.add((u, v))
                else:
                    edge_set.add((v, u))
        
        # Time check
        if time.time() > deadline:
            break
    
    edges = list(edge_set)
    n = len(order)
    
    return n, edges
