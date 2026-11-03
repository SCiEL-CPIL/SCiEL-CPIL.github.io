import time
import sys

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    q = k
    # Determine the smallest d such that the girth of CD(d, q) is at least g.
    # Girth is approx d+5 for odd d, d+6 for even d.
    # We want the smallest d.
    # Try d from 3 upwards.
    best_d = None
    for d in range(3, 20):
        # Check if d+5 (odd) or d+6 (even) >= g
        if d % 2 == 1:
            if d + 5 >= g:
                best_d = d
                break
        else:
            if d + 6 >= g:
                best_d = d
                break
                
    if best_d is None:
        best_d = max(3, g - 5)
        
    d = best_d
    
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
