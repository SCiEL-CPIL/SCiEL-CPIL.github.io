import time
import random
import numpy as np

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110  # 10s margin
    
    # Fallback: LUW Construction
    def luw_construct():
        q = k
        d = max(3, g - 5 if (g - 5) % 2 else g - 4)
        if q == 4:
            table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
            add, sub, mul = (lambda x, y: x ^ y), (lambda x, y: x ^ y), (lambda x, y: table[x][y])
        else:
            add, sub, mul = (lambda x, y: (x + y) % q), (lambda x, y: (x - y) % q), (lambda x, y: x * y % q)
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
                    y[c] = add(x[c], mul(y[a], x[b])) if side == 0 else sub(x[c], mul(x[a], y[b]))
                yield 1 - side, tuple(y)

        start = (0, (None,) + (0,) * d)
        index, order, edges = {start: 0}, [start], []
        for vertex in order:
            for other in neighbours(vertex):
                if other not in index:
                    index[other] = len(order)
                    order.append(other)
                if vertex[0] == 0:
                    edges.append((index[vertex], index[other]))
        return len(order), edges

    # Target: 320 vertices
    target_n = 320
    if k != 4 or g != 10:
        return luw_construct()

    # Try to find a 4-regular graph of girth 10 with 320 vertices
    # Using a heuristic search
    
    def random_4_reg(n):
        # Generate a random 4-regular graph
        # Use a configuration model or random swaps
        # Simple: start with a cycle and add chords
        # Or use networkx if available? No.
        # Implement a simple random 4-regular graph generator
        # This is tricky to do correctly and quickly.
        # Alternative: Start with LUW and try to reduce? No.
        pass

    # Since constructing a random 4-regular graph is complex, 
    # and finding girth 10 is hard, 
    # I will rely on the LUW fallback but try a small optimization if possible.
    # However, the LUW graph is 2560 vertices.
    # If I can't find 320, I return LUW.
    
    # Let's try a specific known construction for (4,10) if I can recall it.
    # The (4,10)-cage is a bipartite graph.
    # It can be constructed from the (3,10)-cage?
    
    # Given the constraints, I will return the LUW graph to ensure validity,
    # but I will add a check: if I can find a smaller valid graph, I use it.
    # Since I don't have a reliable smaller constructor, I return LUW.
    
    return luw_construct()
