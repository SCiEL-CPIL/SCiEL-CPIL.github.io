"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct(k, g):
    """
    Construct a k-regular graph with girth at least g using the
    Lazebnik-Ustimenko-Woldar (LUW) construction CD(d, q).
    
    The LUW construction gives a q-regular bipartite graph with girth
    at least d+5 (for odd d) or d+6 (for even d). We choose d such that
    the girth requirement is met, and q = k (must be prime or 4).
    
    Returns (n, edges) where n is the number of vertices and edges is a
    list of (u, v) pairs.
    """
    start_time = time.time()
    deadline = start_time + 115  # Safety margin: must finish within 120s
    
    q = k
    # For LUW: girth >= d + 5 if d is odd, d + 6 if d is even
    # We need girth >= g, so we need d + 5 >= g (odd d) or d + 6 >= g (even d)
    # Choose smallest valid d
    # If g - 5 is odd, use d = g - 5 (odd), girth = g
    # If g - 5 is even, use d = g - 4 (even), girth = g
    if (g - 5) % 2 == 1:
        d = g - 5
    else:
        d = g - 4
    d = max(3, d)
    
    # Field operations
    if q == 4:
        # GF(4) with representation {0,1,2,3} where 2 = alpha, 3 = alpha+1
        # alpha^2 = alpha + 1, so:
        # 0+0=0, 0+1=1, 0+2=2, 0+3=3
        # 1+1=0, 1+2=3, 1+3=2
        # 2+2=1, 2+3=0
        # 3+3=2
        # Multiplication:
        # 0*x=0, 1*x=x, 2*2=3, 2*3=2, 3*3=1
        mult_table = [
            [0, 0, 0, 0],
            [0, 1, 2, 3],
            [0, 2, 3, 1],
            [0, 3, 1, 2]
        ]
        add_table = [
            [0, 1, 2, 3],
            [1, 0, 3, 2],
            [2, 3, 0, 1],
            [3, 2, 1, 0]
        ]
        def add(x, y):
            return add_table[x][y]
        def sub(x, y):
            # In GF(4), subtraction = addition (characteristic 2)
            return add_table[x][y]
        def mul(x, y):
            return mult_table[x][y]
    else:
        # GF(q) for prime q
        def add(x, y):
            return (x + y) % q
        def sub(x, y):
            return (x - y) % q
        def mul(x, y):
            return (x * y) % q
    
    # Precompute rule for LUW construction
    # rule[c] = (a, b) such that for c = 2..d:
    #   line[c] - point[c] = line[a] * point[b]  (for point-line incidence)
    rule = [None] * (d + 1)
    rule[2] = (1, 1)
    rule[3] = (2, 1)
    rule[4] = (1, 2)
    for c in range(5, d + 1):
        r = (c - 5) % 4
        c0 = c - r
        if r == 0:
            rule[c] = (1, c0 - 2)
        elif r == 1:
            rule[c] = (c0 - 1, 1)
        elif r == 2:
            rule[c] = (c0, 1)
        else:
            rule[c] = (1, c0 + 1)
    
    # BFS from the zero point to build the connected component
    # Vertices are (side, coords) where side is 0 (point) or 1 (line)
    # coords is a tuple of length d+1 (indices 0..d), with coords[0] unused (None)
    
    def get_neighbours(side, x):
        """Yield all neighbours of vertex (side, x)."""
        for first in range(q):
            y = [0] * (d + 1)
            y[0] = 0  # placeholder
            y[1] = first
            if side == 0:
                # x is a point, find all lines y such that y ~ x
                # For c = 2..d: y[c] = x[c] + y[a] * x[b]  where (a,b) = rule[c]
                for c in range(2, d + 1):
                    a, b = rule[c]
                    y[c] = add(x[c], mul(y[a], x[b]))
            else:
                # x is a line, find all points y such that y ~ x
                # For c = 2..d: y[c] = x[c] - x[a] * y[b]  where (a,b) = rule[c]
                for c in range(2, d + 1):
                    a, b = rule[c]
                    y[c] = sub(x[c], mul(x[a], y[b]))
            yield 1 - side, tuple(y)
    
    start = (0, (0,) * (d + 1))
    index = {start: 0}
    order = [start]
    edges = []
    
    for vertex in order:
        if time.time() > deadline:
            break
        side, x = vertex
        for other in get_neighbours(side, x):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if side == 0:
                edges.append((index[vertex], index[other]))
    
    return len(order), edges
# EVOLVE-BLOCK-END
