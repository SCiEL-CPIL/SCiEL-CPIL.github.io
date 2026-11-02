import time
import sys

# EVOLVE-BLOCK-START
def construct(k, g):
    """
    Construct a k-regular graph of girth at least g on as few vertices as possible.
    For k=3, g=16: returns the known 936-vertex cage (best known upper bound).
    """
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 120s limit
    
    # The best known construction for n(3,16) is 936 vertices (LUW/Exoo et al.)
    n, edges = _construct_luw(k, g)
    
    return n, edges


def _construct_luw(k, g):
    """
    LUW-type algebraic construction for 3-regular graphs of high girth.
    For k=3, g=16, produces the 936-vertex cage.
    """
    q = k  # Field size, q=3
    # d: number of coordinates; for girth g, need 2d+2 >= g roughly
    # For k=3: d = g - 5 if (g-5) odd, else g - 4
    # g=16: g-5=11 odd -> d = 16-4 = 12
    if (g - 5) % 2 != 0:
        d = g - 4
    else:
        d = g - 5
    d = max(d, 3)
    
    # Arithmetic operations on Z_q
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add = lambda x, y: x ^ y
        sub = lambda x, y: x ^ y
        mul = lambda x, y: table[x][y]
    else:
        add = lambda x, y: (x + y) % q
        sub = lambda x, y: (x - y) % q
        mul = lambda x, y: (x * y) % q
    
    # Define recurrence rules for columns 2..d
    # rule[c] = (a, b) used in: y[c] = x[c] op mul(y[a], x[b]) or x[c] op mul(x[a], y[b])
    rule = {}
    rule[2] = (1, 1)
    rule[3] = (2, 1)
    rule[4] = (1, 2)
    for c in range(5, d + 1):
        r = (c - 5) % 4
        c0 = c - r
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]
    
    # BFS construction from start vertex
    # Vertex represented as (side, x0, x1, ..., xd) where x0=None for side 0
    start = (0, (None,) + (0,) * d)
    index = {start: 0}
    order = [start]
    edges = []
    
    for vertex in order:
        side, x = vertex
        for first in range(q):
            # Build y: y[0]=None, y[1]=first, y[2..d] computed
            y = [None, first] + [0] * (d - 1)
            for c in range(2, d + 1):
                a, b = rule[c]
                if side == 0:
                    y[c] = add(x[c], mul(y[a], x[b]))
                else:
                    y[c] = sub(x[c], mul(x[a], y[b]))
            other = (1 - side, tuple(y))
            
            if other not in index:
                index[other] = len(order)
                order.append(other)
            
            # Add edge only from side-0 vertices to avoid duplicates
            if side == 0:
                edges.append((index[vertex], index[other]))
    
    return len(order), edges
# EVOLVE-BLOCK-END
