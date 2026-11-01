import time
import random
import itertools
import math

# EVOLVE-BLOCK-START
def construct(k, g):
    start_time = time.time()
    deadline = start_time + 115.0

    if (k, g) == (3, 5):
        return _petersen()
    elif (k, g) == (3, 6):
        return _heawood()
    
    return _luw_construct(k, g)

def _petersen():
    edges = []
    for i in range(5):
        edges.append((i, (i + 1) % 5))
    for i in range(5):
        edges.append((5 + i, 5 + (i + 2) % 5))
    for i in range(5):
        edges.append((i, 5 + i))
    return 10, edges

def _heawood():
    lines = [
        (0, 1, 2),
        (0, 3, 4),
        (0, 5, 6),
        (1, 3, 5),
        (1, 4, 6),
        (2, 3, 6),
        (2, 4, 5)
    ]
    edges = []
    for i, line in enumerate(lines):
        line_id = 7 + i
        for p in line:
            edges.append((p, line_id))
    return 14, edges

def _luw_construct(k, g):
    q = k
    # Minimal d for LUW to guarantee girth >= g
    if g % 2 == 1:
        d = (g - 1) // 2
    else:
        d = g // 2
    d = max(3, d)
    
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
                if side == 0:
                    y[c] = add(x[c], mul(y[a], x[b]))
                else:
                    y[c] = sub(x[c], mul(x[a], y[b]))
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
# EVOLVE-BLOCK-END
