import time
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 115

    # Fallback: LUW construction for all (k, g)
    def luw_construction():
        q = k
        # d is the length of the voltage assignment vector
        # For k=4, g=10, the seed used d=5
        d = max(3, g - 5 if (g - 5) % 2 else g - 4)
        if q == 4:
            # GF(4) operations
            # 0, 1, 2, 3 where 2=alpha, 3=alpha+1
            # alpha^2 = alpha + 1 = 3
            # alpha^3 = alpha(alpha+1) = alpha^2+alpha = 3+2 = 1 (in GF(4), 3+2 = 1? No, 3+2 = 11_2 = 10_2 = 2? 
            # Let's use the table provided in the seed.
            table = [
                [0, 1, 2, 3],
                [1, 2, 3, 0],
                [2, 3, 0, 1],
                [3, 0, 1, 2]
            ]
            # Wait, the seed table was:
            # [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
            # This looks like multiplication in GF(4) if 0 is 0.
            # 1*1=1, 1*2=2, 1*3=3
            # 2*2=3, 2*3=1
            # 3*3=2
            # This is consistent with GF(4).
            
            add = lambda x, y: x ^ y
            sub = lambda x, y: x ^ y # In GF(4), subtraction is same as addition
            mul = lambda x, y: table[x][y]
        else:
            add = lambda x, y: (x + y) % q
            sub = lambda x, y: (x - y) % q
            mul = lambda x, y: (x * y) % q

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
        
        # BFS to construct the graph
        for i, vertex in enumerate(order):
            for other in neighbours(vertex):
                if other not in index:
                    index[other] = len(order)
                    order.append(other)
                if vertex[0] == 0:
                    edges.append((index[vertex], index[other]))
        
        return len(order), edges

    return luw_construction()
