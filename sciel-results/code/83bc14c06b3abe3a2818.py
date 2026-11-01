"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct(k, g):
    # Lazebnik-Ustimenko-Woldar graph CD(d, q) with q = k: an explicit q-regular bipartite graph
    # whose girth is 2d + 2. Points and lines are vectors over GF(q) (q prime, or q = 4).
    # We choose the smallest d such that 2d + 2 >= g.
    
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    q = k
    # Minimum d for LUW graph to have girth >= g
    # Girth = 2d + 2
    # 2d + 2 >= g => d >= (g - 2) / 2
    d = (g - 2 + 1) // 2  # Ceiling of (g-2)/2
    if d < 1:
        d = 1
        
    # Check if we have time to build. LUW construction is O(q^d * d).
    # For q=4, d=4, q^d = 256. Very fast.
    # For q=4, d=5, q^d = 1024. Fast.
    
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add, sub, mul = (lambda x, y: x ^ y), (lambda x, y: x ^ y), (lambda x, y: table[x][y])
    else:
        add, sub, mul = (lambda x, y: (x + y) % q), (lambda x, y: (x - y) % q), (lambda x, y: x * y % q)
        
    rule = {}
    # The rule for LUW graph CD(d, q)
    # For c in 2..d:
    # y[c] = x[c] + y[a]*x[b] (for points)
    # The specific a, b depend on c.
    # Standard LUW uses:
    # c=2: a=1, b=1? No, indices are 1..d.
    # Let's use the rule from the seed program, which is correct.
    
    # Seed program rule:
    # rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    # for c in range(5, d + 1):
    #     r, c0 = (c - 5) % 4, c - (c - 5) % 4
    #     rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]
    
    # This rule generates a graph with girth 2d+2.
    
    if d >= 2:
        rule[2] = (1, 1)
    if d >= 3:
        rule[3] = (2, 1)
    if d >= 4:
        rule[4] = (1, 2)
    for c in range(5, d + 1):
        r, c0 = (c - 5) % 4, c - (c - 5) % 4
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]

    def neighbours(vertex):
        side, x = vertex
        for first in range(q):
            y = [None, first] + [0] * (d - 1)
            for c in range(2, d + 1):
                a, b = rule[c]
                y[c] = add(x[c], mul(y[a], x[b])) if side == 0 else sub(x[c], mul(x[a], y[b]))
            yield 1 - side, tuple(y)

    start = (0, (None,) + (0,) * d)
    index, order, edges = {start: 0}, [start], []
    
    # BFS to build the graph
    for vertex in order:
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
                
    return len(order), edges
# EVOLVE-BLOCK-END
