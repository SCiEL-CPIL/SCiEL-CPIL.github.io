"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
# EVOLVE-BLOCK-START
def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110.0 # 10s margin
    
    # Lazebnik-Ustimenko-Woldar graph CD(d, q)
    # Girth is 2d + 2.
    # We need 2d + 2 >= g => d >= (g-2)/2
    q = k
    d = (g - 2 + 1) // 2  # ceil((g-2)/2)
    
    # Ensure d is at least 1
    if d < 1:
        d = 1

    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add, sub, mul = (lambda x, y: x ^ y), (lambda x, y: x ^ y), (lambda x, y: table[x][y])
    else:
        add, sub, mul = (lambda x, y: (x + y) % q), (lambda x, y: (x - y) % q), (lambda x, y: x * y % q)
        
    # The rule for LUW CD(d,q)
    # The connection between point p and line l is defined by l[c] - p[c] = l[a] * p[b]
    # The indices a, b depend on c.
    # Standard LUW uses a specific sequence.
    # The seed code had a complex rule generation. 
    # For CD(d,q), the rule is often:
    # For c=2..d:
    #   if c is even: a=1, b=c-1
    #   if c is odd: a=c-1, b=1
    # Or similar.
    # Let's stick to the seed's rule logic but ensure it's correct for the girth.
    # The seed's rule:
    # rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    # for c in 5..d: ...
    
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
                # point p ~ line l  iff  l[c] - p[c] = l[a] * p[b]
                # side 0 is point, side 1 is line
                if side == 0:
                    # p=x, l=y
                    # y[c] - x[c] = y[a] * x[b]
                    # y[c] = x[c] + y[a] * x[b]
                    y[c] = add(x[c], mul(y[a], x[b]))
                else:
                    # l=x, p=y
                    # x[c] - y[c] = x[a] * y[b]
                    # y[c] = x[c] - x[a] * y[b]
                    y[c] = sub(x[c], mul(x[a], y[b]))
            yield 1 - side, tuple(y)

    start = (0, (None,) + (0,) * d)
    index, order, edges = {start: 0}, [start], []
    
    # BFS
    for vertex in order:
        if time.time() > deadline:
            break
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
                
    return len(order), edges
# EVOLVE-BLOCK-END
