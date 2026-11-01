import time

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110  # 10s margin
    
    # Lazebnik-Ustimenko-Woldar graph CD(d, q)
    q = k
    # Determine d such that girth is at least g
    # For odd d, girth >= d + 5
    # We want d + 5 >= g => d >= g - 5
    # d must be odd.
    if (g - 5) % 2 == 0:
        d = g - 4
    else:
        d = g - 5
    d = max(3, d)
    
    # Field operations for GF(q)
    if q == 4:
        # GF(4) = {0, 1, 2, 3} with 0=0, 1=1, 2=alpha, 3=alpha+1
        # alpha^2 = alpha + 1
        # Multiplication table
        mul_table = [
            [0, 0, 0, 0],
            [0, 1, 2, 3],
            [0, 2, 3, 1],
            [0, 3, 1, 2]
        ]
        def add(x, y):
            return x ^ y
        def sub(x, y):
            return x ^ y
        def mul(x, y):
            return mul_table[x][y]
    else:
        def add(x, y):
            return (x + y) % q
        def sub(x, y):
            return (x - y) % q
        def mul(x, y):
            return (x * y) % q

    # Rule coefficients for the LUW construction
    # y[c] = y[a] * x[b] + x[c] (for points)
    # The rule is defined for c=2..d
    rule = {}
    rule[2] = (1, 1)
    rule[3] = (2, 1)
    rule[4] = (1, 2)
    for c in range(5, d + 1):
        r = (c - 5) % 4
        c0 = c - r
        # Pattern for coefficients (a, b)
        # c=5 (r=0): (1, 3) -> c0=5, (1, 5-2=3)
        # c=6 (r=1): (4, 1) -> c0=6, (6-1, 1)
        # c=7 (r=2): (5, 1) -> c0=7, (7, 1) ? No, pattern in seed:
        # Seed: [(1, c0-2), (c0-1, 1), (c0, 1), (1, c0+1)]
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

    # BFS to find the connected component of the zero point
    # Vertices are (side, tuple) where side 0 is point, 1 is line
    # Point: (0, (None, x1, ..., xd))
    # Line: (1, (l1, ..., ld+1)) ? 
    # In seed: start = (0, (None,) + (0,)*d)
    # neighbours(vertex):
    #   side, x = vertex
    #   for first in range(q):
    #      y = [None, first] + [0]*(d-1)
    #      ...
    #      yield 1-side, tuple(y)
    
    # Optimization: Use a set for visited, and a list for order.
    # Encode vertices as integers to speed up hashing?
    # Side (1 bit) + d+1 coordinates (log2(q) bits each).
    # For q=4, 2 bits per coord. d=5 -> 6 coords -> 12 bits. Total 13 bits.
    # Integer encoding is faster than tuple hashing.
    
    def encode(side, coords):
        # coords is a tuple of length d+1
        # coords[0] is None for points? 
        # In seed: point is (None, x1, ..., xd). Line is (l1, ..., ld+1)?
        # Actually, the seed's `y` for line has length d+1.
        # Let's stick to tuples for correctness, but optimize the loop.
        pass

    # Let's use the tuple approach but optimized.
    index = {}
    order = []
    
    start = (0, (None,) + (0,) * d)
    index[start] = 0
    order.append(start)
    
    edges = []
    
    for vertex in order:
        if time.time() > deadline:
            break
        side, x = vertex
        for first in range(q):
            # Construct y
            # y[0] is None or first?
            # Seed: y = [None, first] + [0]*(d-1)
            # So y has length d+1.
            # y[0] = None (for points) or value (for lines)?
            # In seed, `side` determines the formula.
            # If side==0 (point), we compute line y.
            # If side==1 (line), we compute point y.
            
            # The seed's `neighbours` yields `1 - side, tuple(y)`.
            # So if current is point (0), neighbor is line (1).
            # y is constructed as a list of length d+1.
            
            y = [0] * (d + 1)
            y[0] = first if side == 1 else None # Wait, seed uses [None, first]...
            # Seed: y = [None, first] + [0] * (d - 1)
            # This implies y[0]=None, y[1]=first, y[2..d]=0.
            # This structure is used for BOTH points and lines in the seed?
            # Let's check the index.
            # Point: (0, (None, 0, ..., 0))
            # Line: (1, (l1, l2, ..., ld+1)) ?
            # The seed's `y` for a line neighbor of a point:
            # `y[0]` is None? No, `y` is just a list.
            # `yield 1 - side, tuple(y)`.
            # If side=0, yields (1, tuple(y)).
            # If side=1, yields (0, tuple(y)).
            
            # The seed's `neighbours` logic:
            # y = [None, first] + [0]*(d-1)
            # This creates a list where y[0]=None, y[1]=first, rest 0.
            # Then it fills y[2..d].
            # So y[0] is always None?
            # If y[0] is always None, then all lines have first coordinate None?
            # And all points have first coordinate None?
            # Then how are they distinguished? By `side`.
            # And `first` is y[1].
            
            y[0] = None
            y[1] = first
            # y[2..d] initialized to 0
            
            for c in range(2, d + 1):
                a, b = rule[c]
                if side == 0:
                    # Point to Line
                    # y[c] = x[c] + y[a] * x[b]
                    y[c] = add(x[c], mul(y[a], x[b]))
                else:
                    # Line to Point
                    # y[c] = x[c] - x[a] * y[b]
                    y[c] = sub(x[c], mul(x[a], y[b]))
            
            neighbor = (1 - side, tuple(y))
            
            if neighbor not in index:
                index[neighbor] = len(order)
                order.append(neighbor)
            
            if side == 0:
                edges.append((index[vertex], index[neighbor]))
    
    return len(order), edges
