"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import sys

# EVOLVE-BLOCK-START
def construct(k, g):
    """
    Constructs a k-regular graph of girth at least g.
    Uses hardcoded small cages for known small (k,g) pairs.
    Falls back to Lazebnik-Ustimenko-Woldar (L-U-W) construction for general cases.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # 10 second safety margin
    
    # Hardcoded solutions for small, well-known cages
    # Format: (k, g): (n, edges)
    hardcoded = {}
    
    # Petersen Graph: k=3, g=5, n=10
    hardcoded[(3, 5)] = (10, [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0),
                              (0, 5), (1, 6), (2, 7), (3, 8), (4, 9),
                              (5, 8), (8, 6), (6, 9), (9, 7), (7, 5)])
    
    # k=4, g=6, n=42 (Complement of Petersen? No, specific cage)
    # The 42-vertex cage is the graph of the 4-dimensional hypercube Q4? No, Q4 is k=4, g=4.
    # The 42-vertex 4-regular girth 6 graph is known.
    # I will not hardcode complex edge lists to avoid errors, unless I'm sure.
    # For now, rely on L-U-W for k=4.
    
    if (k, g) in hardcoded:
        return hardcoded[(k, g)]
    
    # L-U-W Construction
    q = k
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add = lambda x, y: x ^ y
        sub = lambda x, y: x ^ y
        mul = lambda x, y: table[x][y]
    elif q == 2:
        add = lambda x, y: x ^ y
        sub = lambda x, y: x ^ y
        mul = lambda x, y: x & y
    else:
        add = lambda x, y: (x + y) % q
        sub = lambda x, y: (x - y) % q
        mul = lambda x, y: x * y % q

    # Rule for L-U-W: l[c] - p[c] = l[a] * p[b]
    # For c=2..d
    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r = (c - 5) % 4
        c0 = c - (c - 5) % 4
        # Indices in the list correspond to r=0,1,2,3
        # r=0: (1, c0-2), r=1: (c0-1, 1), r=2: (c0, 1), r=3: (1, c0+1)
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]

    def neighbours(vertex):
        side, x = vertex
        for first in range(q):
            y = [None, first] + [0] * (d - 1)
            for c in range(2, d + 1):
                a, b = rule[c]
                if side == 0:
                    # Point to Line: y[c] = x[c] + y[a]*x[b]
                    y[c] = add(x[c], mul(y[a], x[b]))
                else:
                    # Line to Point: x[c] = y[c] + x[a]*y[b] => y[c] = x[c] - x[a]*y[b]
                    y[c] = sub(x[c], mul(x[a], y[b]))
            yield 1 - side, tuple(y)

    start = (0, (None,) + (0,) * d)
    index = {start: 0}
    order = [start]
    edges = []
    
    # BFS/DFS to build graph
    for i, vertex in enumerate(order):
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
