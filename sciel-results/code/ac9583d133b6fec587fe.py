"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import sys

# EVOLVE-BLOCK-START
def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s kill
    
    # Lazebnik-Ustimenko-Woldar graph CD(d, q)
    # We require degree k. LUW graphs are q-regular.
    # So we set q = k. This works if k is a prime power.
    # For k=5, this is valid.
    q = k
    
    if q != 5:
        # Fallback for other k if needed, but task is specific to k=5.
        # If k is not 5, we might not have a good construction.
        # We will try to use LUW if q is a prime power, else random.
        pass
        
    # Calculate minimal d for girth g
    # LUW graph CD(d, q) has girth at least 2d + 1
    # We need 2d + 1 >= g  => d >= (g-1)/2
    d = (g + 1) // 2  # Ceil((g-1)/2)
    # Ensure d is at least 1
    d = max(1, d)
    
    # For q=5, d=4 => n = 2 * 5^4 = 1250
    # For q=5, d=5 => n = 2 * 5^5 = 6250
    
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add, sub, mul = (lambda x, y: x ^ y), (lambda x, y: x ^ y), (lambda x, y: table[x][y])
    else:
        # Check if q is prime? For small q it's fine.
        add, sub, mul = (lambda x, y: (x + y) % q), (lambda x, y: (x - y) % q), (lambda x, y: x * y % q)
        
    # Rule generation for LUW
    # The rule maps c to (a, b)
    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r, c0 = (c - 5) % 4, c - (c - 5) % 4
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]

    def neighbours(vertex):
        side, x = vertex
        # x is a tuple of length d+1? 
        # In seed: y = [None, first] + [0] * (d - 1)
        # So x has indices 1..d?
        # Seed: start = (0, (None,) + (0,) * d)
        # So x[1]..x[d] are the coordinates.
        for first in range(q):
            y = [None, first] + [0] * (d - 1)
            for c in range(2, d + 1):
                a, b = rule[c]
                # point p ~ line l  iff  l[c] - p[c] = l[a] * p[b]
                if side == 0:
                    # p is point, l is line
                    # y is line, x is point
                    # y[c] - x[c] = y[a] * x[b] ?
                    # Seed: y[c] = add(x[c], mul(y[a], x[b]))
                    y[c] = add(x[c], mul(y[a], x[b]))
                else:
                    # p is line, l is point?
                    # Seed: y[c] = sub(x[c], mul(x[a], y[b]))
                    y[c] = sub(x[c], mul(x[a], y[b]))
            yield 1 - side, tuple(y)

    start = (0, (None,) + (0,) * d)
    index = {start: 0}
    order = [start]
    edges = []
    
    # BFS to generate the connected component
    # Note: LUW graphs are connected.
    i = 0
    while i < len(order):
        vertex = order[i]
        i += 1
        
        # Time check
        if time.time() > deadline:
            # Return whatever we have? 
            # If we don't have a full graph, it might be invalid.
            # But LUW is deterministic and fast.
            # 1250 vertices, 5 edges each. ~6250 iterations.
            # This is very fast.
            pass
            
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))

    n = len(order)
    # Verify it's a simple graph? 
    # LUW construction guarantees simplicity and regularity.
    return n, edges

# EVOLVE-BLOCK-END
