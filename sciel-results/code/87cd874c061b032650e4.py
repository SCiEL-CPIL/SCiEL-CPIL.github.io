import time
import random

def construct(k, g):
    start_time = time.time()
    deadline = 110.0  # Safety margin before 120s limit
    
    # Known small cages
    if k == 3 and g == 5:
        return petersen_graph()
    if k == 3 and g == 6:
        return heawood_graph()
    if k == 3 and g == 7:
        return cage_3_7()
    if k == 3 and g == 8:
        return cage_3_8()
    if k == 3 and g == 9:
        return cage_3_9()
    if k == 3 and g == 10:
        return cage_3_10()
    if k == 3 and g == 11:
        return cage_3_11()
    if k == 3 and g == 12:
        return cage_3_12()
    if k == 3 and g == 13:
        return cage_3_13()
    if k == 3 and g == 14:
        return cage_3_14()
    if k == 3 and g == 15:
        return cage_3_15()
    if k == 3 and g == 16:
        return cage_3_16()
    if k == 3 and g == 17:
        return cage_3_17()
    if k == 3 and g == 18:
        return cage_3_18()
    if k == 3 and g == 19:
        return cage_3_19()
    if k == 3 and g == 20:
        return cage_3_20()
    
    # LUW construction
    return luw_construct(k, g, deadline)

def petersen_graph():
    n = 10
    edges = []
    # Outer pentagon
    for i in range(5):
        edges.append((i, (i + 1) % 5))
    # Inner star
    for i in range(5):
        edges.append((5 + i, 5 + (i + 2) % 5))
    # Spokes
    for i in range(5):
        edges.append((i, 5 + i))
    return n, edges

def heawood_graph():
    n = 14
    edges = []
    # Heawood graph is the incidence graph of the Fano plane
    # 7 points, 7 lines
    # Points: 0..6, Lines: 7..13
    # Fano plane lines:
    # 012, 034, 056, 135, 146, 236, 245
    lines = [
        (0, 1, 2),
        (0, 3, 4),
        (0, 5, 6),
        (1, 3, 5),
        (1, 4, 6),
        (2, 3, 6),
        (2, 4, 5)
    ]
    for i, line in enumerate(lines):
        for p in line:
            edges.append((p, 7 + i))
    return n, edges

def cage_3_7():
    # 18 vertices, 3-regular, girth 7
    # This is the incidence graph of the (3,7)-cage
    # Can be constructed from the Petersen graph by expanding vertices?
    # Or use a known construction.
    # For now, fall back to LUW if not implemented
    return luw_construct(3, 7, time.time() + 100)

def cage_3_8():
    return luw_construct(3, 8, time.time() + 100)

def cage_3_9():
    return luw_construct(3, 9, time.time() + 100)

def cage_3_10():
    return luw_construct(3, 10, time.time() + 100)

def cage_3_11():
    return luw_construct(3, 11, time.time() + 100)

def cage_3_12():
    return luw_construct(3, 12, time.time() + 100)

def cage_3_13():
    return luw_construct(3, 13, time.time() + 100)

def cage_3_14():
    return luw_construct(3, 14, time.time() + 100)

def cage_3_15():
    return luw_construct(3, 15, time.time() + 100)

def cage_3_16():
    return luw_construct(3, 16, time.time() + 100)

def cage_3_17():
    return luw_construct(3, 17, time.time() + 100)

def cage_3_18():
    return luw_construct(3, 18, time.time() + 100)

def cage_3_19():
    return luw_construct(3, 19, time.time() + 100)

def cage_3_20():
    return luw_construct(3, 20, time.time() + 100)

def luw_construct(k, g, deadline):
    q = k
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    
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
