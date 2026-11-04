"""
Constructs a k-regular graph of girth g.
Uses the Lazebnik-Ustimenko-Woldar (L-U-W) construction for k=4, g=10.
Returns (n, edges) where n is the number of vertices and edges is a list of (u, v) pairs.
"""

def construct(k, g):
    # The L-U-W construction CD(d, q)
    # For k=4, we use q=4.
    # We need girth >= g.
    # The girth of CD(d, q) is at least d + 5 (if d is odd) or d + 4 (if d is even)?
    # Actually, the girth is 2 * (d + 1) for some cases?
    # Let's stick to the seed's logic which worked.
    
    q = k
    # Determine d such that the girth is at least g.
    # Seed logic: d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    # For g=10: d = 5.
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    
    # GF(q) operations
    if q == 4:
        # GF(4) is isomorphic to Z_2 x Z_2. We can use XOR for addition/subtraction.
        # Multiplication table for GF(4) elements {0, 1, 2, 3}
        # 0 is identity for add, 0 for mul.
        # 1 is identity for mul.
        # 2*2=3, 2*3=1, 3*3=2.
        table = [
            [0, 0, 0, 0],
            [0, 1, 2, 3],
            [0, 2, 3, 1],
            [0, 3, 1, 2]
        ]
        def add(x, y): return x ^ y
        def sub(x, y): return x ^ y
        def mul(x, y): return table[x][y]
    else:
        # Assume q is prime for simplicity, though L-U-W works for prime powers.
        # For k=4, q=4 is the only case.
        def add(x, y): return (x + y) % q
        def sub(x, y): return (x - y) % q
        def mul(x, y): return (x * y) % q

    # Precompute the rule coefficients for the L-U-W graph
    # rule[c] = (a, b) such that l[c] - p[c] = l[a] * p[b]
    rule = {}
    # The rule depends on c and d.
    # From the seed:
    # rule[2] = (1, 1)
    # rule[3] = (2, 1)
    # rule[4] = (1, 2)
    # For c > 4:
    # r, c0 = (c - 5) % 4, c - (c - 5) % 4
    # rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]
    
    for c in range(2, d + 1):
        if c == 2:
            rule[c] = (1, 1)
        elif c == 3:
            rule[c] = (2, 1)
        elif c == 4:
            rule[c] = (1, 2)
        else:
            r = (c - 5) % 4
            c0 = c - r
            # The list of 4 options
            opts = [
                (1, c0 - 2),
                (c0 - 1, 1),
                (c0, 1),
                (1, c0 + 1)
            ]
            rule[c] = opts[r]

    def neighbours(vertex):
        side, x = vertex
        # x is a tuple of length d+1? 
        # Seed: y = [None, first] + [0] * (d - 1) -> length 1 + 1 + d-1 = d+1.
        # Indices 0..d. Index 0 is None. Index 1 is 'first'. Indices 2..d are the rest.
        # Wait, the seed code:
        # y = [None, first] + [0] * (d - 1)
        # Length is 2 + d - 1 = d + 1.
        # Indices: 0, 1, ..., d.
        # x[c] is used for c in 2..d.
        
        for first in range(q):
            y = [None] * (d + 1)
            y[0] = None
            y[1] = first
            for c in range(2, d + 1):
                y[c] = 0 # Initialize, will be overwritten
            
            # Calculate y[2..d]
            for c in range(2, d + 1):
                a, b = rule[c]
                if side == 0:
                    # Point p ~ Line l
                    # l[c] - p[c] = l[a] * p[b]
                    # y[c] = p[c] + l[a] * p[b]
                    # Here y is the line, x is the point.
                    y[c] = add(x[c], mul(y[a], x[b]))
                else:
                    # Line l ~ Point p
                    # l[c] - p[c] = l[a] * p[b]
                    # p[c] = l[c] - l[a] * p[b]
                    # Here x is the line, y is the point.
                    y[c] = sub(x[c], mul(x[a], y[b]))
            
            yield (1 - side, tuple(y))

    # Start from zero point
    start = (0, (None,) + (0,) * d)
    
    index = {start: 0}
    order = [start]
    edges = []
    
    # BFS to find the connected component
    for i in range(len(order)):
        vertex = order[i]
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            
            # Add edge only once (when from point to line, or just ensure unique)
            # The graph is bipartite. 
            # If we iterate over all vertices, we will see each edge twice.
            # We can just add edges where side==0 to avoid duplicates.
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))

    return len(order), edges
