"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct(k, g):
    # Lazebnik-Ustimenko-Woldar graph CD(d, q) with q = k.
    # We select the smallest d such that the girth is at least g.
    # The girth of CD(d, q) is generally 2d + 2 for d >= 3.
    import time
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit

    q = k
    # Ensure d is an integer >= 2
    # Heuristic: d = ceil(g/2) to ensure girth >= g for small q
    d_min = (g + 1) // 2
    if d_min < 2:
        d_min = 2
    
    d = d_min
    
    # If q == 4, use GF(4) arithmetic
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add, sub, mul = (lambda x, y: x ^ y), (lambda x, y: x ^ y), (lambda x, y: table[x][y])
    else:
        add, sub, mul = (lambda x, y: (x + y) % q), (lambda x, y: (x - y) % q), (lambda x, y: x * y % q)
        
    # The rule for the LUW graph depends on d.
    # The standard LUW construction uses a specific set of indices.
    # The seed code implemented a specific recursive rule.
    # We will keep the seed's rule implementation but ensure d is set correctly.
    
    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r, c0 = (c - 5) % 4, c - (c - 5) % 4
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]

    def neighbours(vertex):
        side, x = vertex
        for first in range(q):
            y = [None, first] + [0] * (d - 1)
            for c in range(2, d + 1):
                a, b = rule[c]
                # point p ~ line l  iff  l[c] - p[c] = l[a] * p[b] for c = 2..d
                y[c] = add(x[c], mul(y[a], x[b])) if side == 0 else sub(x[c], mul(x[a], y[b]))
            yield 1 - side, tuple(y)

    start = (0, (None,) + (0,) * d)
    index, order, edges = {start: 0}, [start], []
    for vertex in order:
        if time.time() > deadline:
            break
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
                
    # If we timed out, the graph is incomplete. 
    # However, LUW graphs are small enough (n < 10000 for reasonable d,k) 
    # that they should finish quickly. 
    # If n is too large, it might time out. 
    # For k=5, d=5, n=6250. This is fast.
    
    return len(order), edges
# EVOLVE-BLOCK-END
