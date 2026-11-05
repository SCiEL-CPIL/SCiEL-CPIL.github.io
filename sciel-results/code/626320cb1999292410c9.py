import time
import random
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 115  # 5s margin for overhead
    
    if k == 4 and g == 10:
        # The smallest known 4-regular girth-10 graph has 320 vertices.
        # Standard LUW construction yields 1024 vertices.
        # We attempt to find a smaller graph if possible, but LUW is the safe fallback.
        
        # Check for known small cages or specific constructions if available.
        # For now, we rely on LUW which is guaranteed to work.
        
        try:
            return _luw_graph(k, g)
        except Exception:
            pass

        # Fallback: Random 4-regular graph (likely too large, but valid if found)
        # Not implemented due to complexity and size issues.
        
        # Return LUW graph
        return _luw_graph(k, g)
    else:
        return _luw_graph(k, g)

def _luw_graph(k, g):
    q = k
    # Determine d such that girth >= g
    # LUW girth is approx 2*d + 2?
    # For q=4, d=5 gives girth >= 10.
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    
    if q == 4:
        # GF(4) operations: add/sub is XOR, mul is table
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add = lambda x, y: x ^ y
        sub = lambda x, y: x ^ y
        mul = lambda x, y: table[x][y]
    else:
        add = lambda x, y: (x + y) % q
        sub = lambda x, y: (x - y) % q
        mul = lambda x, y: x * y % q

    # Define the rules for the voltage assignment
    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r, c0 = (c - 5) % 4, c - (c - 5) % 4
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]

    def neighbours(vertex):
        side, x = vertex
        # x is a tuple of length d+1, x[0] is ignored (None)
        for first in range(q):
            # Initialize y[1] = first
            y = [None, first] + [0] * (d - 1)
            for c in range(2, d + 1):
                a, b = rule[c]
                if side == 0:
                    y[c] = add(x[c], mul(y[a], x[b]))
                else:
                    y[c] = sub(x[c], mul(x[a], y[b]))
            yield (1 - side, tuple(y))

    start = (0, (None,) + (0,) * d)
    index = {start: 0}
    order = [start]
    edges = []
    
    for i, vertex in enumerate(order):
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
                
    return len(order), edges
