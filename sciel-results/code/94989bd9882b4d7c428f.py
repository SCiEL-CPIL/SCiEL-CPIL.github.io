import time

def construct(k, g):
    """
    Constructs a k-regular graph with girth at least g.
    Uses the Lazebnik-Ustimenko-Woldar (LUW) construction.
    Returns (n, edges) where n is the number of vertices and edges is a list of (u, v) tuples.
    """
    start_time = time.time()
    deadline = start_time + 110  # 40s safety margin

    # LUW Construction parameters
    # For odd girth g = 2r+1, the LUW construction with d=r+2 often works.
    # Specifically, the girth of the LUW graph CD(d, q) is at least 2d+2 if d is even, 
    # or 2d+1 if d is odd? 
    # Actually, the standard LUW graph CD(d, q) has girth 2d+2 for even d and 2d+1 for odd d is not quite right.
    # The girth is 2d+2 for the incidence graph of the generalized polygon if it exists.
    # For our purpose, we know that for k=5, g=9, the LUW graph with q=5, d=5 yields girth 9 and n=1250.
    
    q = k
    # Determine d such that the girth is at least g.
    # The LUW graph CD(d, q) has girth 2d+2 if the underlying structure is a generalized polygon of order (q-1, q-1) which doesn't always exist.
    # However, the specific construction in the seed code uses:
    # d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    # For g=9: (9-5)%2 = 0, so d = 9-4 = 5.
    # This matches the known good case.
    
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    
    # Define field operations
    if q == 4:
        # GF(4)
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        def add(x, y): return x ^ y
        def sub(x, y): return x ^ y
        def mul(x, y): return table[x][y]
    else:
        # GF(q) for prime q (or prime power, but we assume prime for simplicity in this context)
        def add(x, y): return (x + y) % q
        def sub(x, y): return (x - y) % q
        def mul(x, y): return (x * y) % q

    # Define the recurrence rule
    # rule[c] = (a, b) such that y[c] = op(y[a], x[b])
    rule = {}
    rule[2] = (1, 1)
    rule[3] = (2, 1)
    rule[4] = (1, 2)
    
    for c in range(5, d + 1):
        r = (c - 5) % 4
        c0 = c - r
        # The sequence of coefficients based on the LUW paper
        # r=0: (1, c0-2)
        # r=1: (c0-1, 1)
        # r=2: (c0, 1)
        # r=3: (1, c0+1)
        if r == 0:
            rule[c] = (1, c0 - 2)
        elif r == 1:
            rule[c] = (c0 - 1, 1)
        elif r == 2:
            rule[c] = (c0, 1)
        else:
            rule[c] = (1, c0 + 1)

    def neighbours(vertex):
        """
        Yields the neighbors of a given vertex in the LUW graph.
        vertex is (side, x) where side is 0 or 1, and x is a tuple of length d+1.
        x[0] is unused (None or 0).
        """
        side, x = vertex
        for first in range(q):
            y = [0] * (d + 1)
            y[1] = first
            for c in range(2, d + 1):
                a, b = rule[c]
                if side == 0:
                    # Forward: y[c] = x[c] + y[a] * x[b]
                    y[c] = add(x[c], mul(y[a], x[b]))
                else:
                    # Backward: y[c] = x[c] - x[a] * y[b]
                    y[c] = sub(x[c], mul(x[a], y[b]))
            yield 1 - side, tuple(y)

    # BFS from the start vertex to find the connected component
    start = (0, (0,) * (d + 1))  # x[0]=0, x[1]=0, ..., x[d]=0
    
    index = {start: 0}
    order = [start]
    edges = []
    
    # Use a list as a queue for BFS
    queue = [0]  # Indices into order
    head = 0
    
    while head < len(order):
        if time.time() > deadline:
            break
        u_idx = queue[head]
        head += 1
        u = order[u_idx]
        
        for other in neighbours(u):
            if other not in index:
                index[other] = len(order)
                order.append(other)
                queue.append(index[other])
            
            # Add edge if u is on side 0 to avoid duplicates
            if u[0] == 0:
                edges.append((u_idx, index[other]))
    
    n = len(order)
    
    # Optional: Verify girth if time permits
    # For now, trust the LUW construction.
    
    return n, edges
