import time
import random
import math
from collections import deque

def construct(k, g):
    # Fallback: Lazebnik-Ustimenko-Woldar graph
    def luw_construct(k, g):
        q = k
        # Determine d based on girth
        # For k=4, g=10, d=5 yields n=320 in the specific LUW variant?
        # The formula d = max(3, g - 5 if (g - 5) % 2 else g - 4) is used.
        d = max(3, g - 5 if (g - 5) % 2 else g - 4)
        
        if q == 4:
            # GF(4) operations
            # 0, 1, 2, 3
            # 2 = 1+1, 3 = 1+2
            # 2*2 = 0? No, in GF(4), x^2+x+1=0, so 2^2 = 2+1 = 3?
            # Let's use the table provided in the seed, which is correct for GF(4).
            table = [[0, 0, 0, 0], 
                     [0, 1, 2, 3], 
                     [0, 2, 3, 1], 
                     [0, 3, 1, 2]]
            add = lambda x, y: x ^ y
            sub = lambda x, y: x ^ y
            mul = lambda x, y: table[x][y]
        else:
            # Generic prime field or just modulo q (works if q is prime)
            add = lambda x, y: (x + y) % q
            sub = lambda x, y: (x - y) % q
            mul = lambda x, y: x * y % q
            
        # Define the rules for the LUW construction
        # rule[c] = (a, b) such that y[c] = x[c] +/- y[a]*x[b]
        rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
        for c in range(5, d + 1):
            r, c0 = (c - 5) % 4, c - (c - 5) % 4
            # The seed uses a specific pattern
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
        
        for i in range(len(order)):
            vertex = order[i]
            for other in neighbours(vertex):
                if other not in index:
                    index[other] = len(order)
                    order.append(other)
                if vertex[0] == 0:
                    edges.append((index[vertex], index[other]))
                    
        return len(order), edges

    # For k=4, g=10, the LUW construction is the best known explicit construction
    # The random search in the seed program was likely futile and slow.
    # We return the LUW construction directly.
    return luw_construct(k, g)
