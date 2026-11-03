import time
import random
import numpy as np

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # LUW graph construction: CD(d, q) with q = k
    # Girth = d + 5 for odd d
    # Number of vertices = 2 * q^(d+1)
    def luw_construct(k, g):
        q = k
        # For girth >= g, need d + 5 >= g, so d >= g - 5
        # Use minimal d
        d = g - 5
        if d < 3:
            d = 3
        
        if q == 4:
            table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
            add = lambda x, y: x ^ y
            sub = lambda x, y: x ^ y
            mul = lambda x, y: table[x][y]
        else:
            add = lambda x, y: (x + y) % q
            sub = lambda x, y: (x - y) % q
            mul = lambda x, y: x * y % q
        
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
                        y[c] = add(x[c], mul(y[a], x[b]))
                    else:
                        y[c] = sub(x[c], mul(x[a], y[b]))
                yield 1 - side, tuple(y)
        
        start = (0, (None,) + (0,) * d)
        index = {start: 0}
        order = [start]
        edges = []
        
        for vertex in order:
            for other in neighbours(vertex):
                if other not in index:
                    index[other] = len(order)
                    order.append(other)
                if vertex[0] == 0:
                    edges.append((index[vertex], index[other]))
        
        return len(order), edges
    
    # For k=5, g=9: use LUW with d=4, girth=9
    # This gives 2*5^5 = 6250 vertices
    # Try to find a smaller valid graph if possible
    
    if k == 5 and g == 9:
        # First try a known smaller construction if available
        # For now, use LUW which is guaranteed valid
        n, edges = luw_construct(k, g)
        return n, edges
    else:
        return luw_construct(k, g)
