"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math

def construct(k, g):
    """Construct a k-regular graph with girth at least g.
    
    Uses the Lazebnik-Ustimenko-Woldar (LUW) construction for prime q=k.
    Returns (n, edges) where n is the number of vertices and edges is a list of (i,j) pairs.
    """
    start_time = time.time()
    deadline = 115.0  # safety margin before 120s limit
    
    q = k  # must be prime for the standard LUW construction
    
    # Determine the depth d needed for girth >= g
    # LUW(d, q) has girth >= d + 5 for odd d
    # We need d + 5 >= g, so d >= g - 5
    # d must be odd
    if (g - 5) % 2 == 1:
        d = g - 5
    else:
        d = g - 4  # next odd >= g-5
    
    # Ensure d is at least 3
    d = max(3, d)
    
    # Build the rule table for LUW
    # rule[c] = (a, b) means: y[c] = x[c] + y[a] * x[b] (for point side)
    # or y[c] = x[c] - x[a] * y[b] (for line side)
    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r = (c - 5) % 4
        c0 = c - r
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]
    
    # Field operations for GF(q)
    def add(x, y):
        return (x + y) % q
    
    def sub(x, y):
        return (x - y) % q
    
    def mul(x, y):
        return x * y % q
    
    # BFS to construct the graph
    # Vertices are (side, x) where side is 0 (point) or 1 (line)
    # x is a tuple of length d+1 (indices 0..d, with x[0] being None/unused)
    
    index = {}  # (side, x_tuple) -> vertex_id
    order = []  # list of vertices in BFS order
    edges = []
    
    # Start vertex: point at zero
    start_vertex = (0, (None,) + (0,) * d)
    index[start_vertex] = 0
    order.append(start_vertex)
    
    # BFS
    for vi, vertex in enumerate(order):
        side, x = vertex
        for first in range(q):
            # Build the neighbor
            y = [None, first] + [0] * (d - 1)
            for c in range(2, d + 1):
                a, b = rule[c]
                if side == 0:
                    # point -> line: y[c] = x[c] + y[a] * x[b]
                    y[c] = add(x[c], mul(y[a], x[b]))
                else:
                    # line -> point: y[c] = x[c] - x[a] * y[b]
                    y[c] = sub(x[c], mul(x[a], y[b]))
            
            neighbor = (1 - side, tuple(y))
            
            if neighbor not in index:
                index[neighbor] = len(order)
                order.append(neighbor)
            
            # Add edge only from point side to avoid duplicates
            if side == 0:
                edges.append((vi, index[neighbor]))
        
        # Check time
        if vi % 100 == 0 and time.time() > deadline:
            break
    
    n = len(order)
    return n, edges
# EVOLVE-BLOCK-END
