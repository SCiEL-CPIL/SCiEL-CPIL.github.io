import time
import sys

# EVOLVE-BLOCK-START
def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110  # 110s safety margin for 120s limit

    q = k
    # The girth of the Lazebnik-Ustimenko-Woldar graph CD(d, q) is 2d + 2.
    # We need 2d + 2 >= g, so d >= (g - 2) / 2.
    # Since d must be an integer, d = max(1, (g - 2) // 2).
    # Note: The seed code used a more complex d calculation which often resulted in a larger d.
    d = max(1, (g - 2) // 2)
    
    # Ensure d is at least 1 for the construction to work (CD(1, q) is a cycle? No, CD(1,q) has girth 4? 
    # Actually, the definition of CD(d,q) usually requires d>=1.
    # For d=1, girth is 4.
    # For d=2, girth is 6.
    # For d=3, girth is 8.
    # For d=4, girth is 10.
    
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add = lambda x, y: x ^ y
        sub = lambda x, y: x ^ y
        mul = lambda x, y: table[x][y]
    else:
        add = lambda x, y: (x + y) % q
        sub = lambda x, y: (x - y) % q
        mul = lambda x, y: x * y % q

    # Rule for the LUW construction
    # y[c] = y[c-1] + y[c-2] * ... 
    # The standard rule for CD(d, q) is:
    # p ~ l iff l[c] - p[c] = l[c-1] * p[c-2] ? No.
    # The seed code uses a specific rule:
    # rule[c] = (a, b) such that y[c] = y[c-1] + y[a] * y[b] ?
    # Let's stick to the seed code's rule generation to ensure correctness of the LUW family.
    
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
                if side == 0:
                    # point p ~ line l: l[c] - p[c] = l[a] * p[b]
                    # l[c] = p[c] + l[a] * p[b]
                    y[c] = add(x[c], mul(y[a], x[b]))
                else:
                    # line l ~ point p: l[c] - p[c] = l[a] * p[b]
                    # p[c] = l[c] - l[a] * p[b] ... wait
                    # The relation is symmetric in the bipartite graph construction.
                    # If l[c] = p[c] + l[a] p[b], then p[c] = l[c] - l[a] p[b].
                    y[c] = sub(x[c], mul(x[a], y[b]))
            yield 1 - side, tuple(y)

    start = (0, (None,) + (0,) * d)
    index, order, edges = {start: 0}, [start], []
    
    # BFS to build the graph
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
