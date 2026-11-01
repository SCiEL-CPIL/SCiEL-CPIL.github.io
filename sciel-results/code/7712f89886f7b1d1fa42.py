import time
import random

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110  # 10s margin
    
    # Lazebnik-Ustimenko-Woldar graph CD(d, q) with q = k
    # This is a q-regular bipartite graph.
    # Girth is d+5 if d is odd, d+4 if d is even.
    # We need girth >= g.
    # For g=9, we need girth >= 9.
    # If we pick d such that girth >= 9.
    
    q = k
    
    # Determine d
    # If d is odd, girth = d+5. Need d+5 >= g => d >= g-5.
    # If d is even, girth = d+4. Need d+4 >= g => d >= g-4.
    # We want the smallest d to minimize vertices (n = 2*q^d).
    
    d_candidates = []
    # Try odd d
    d_odd = g - 5
    if d_odd < 3: d_odd = 3
    if d_odd % 2 == 0: d_odd += 1
    d_candidates.append(d_odd)
    
    # Try even d
    d_even = g - 4
    if d_even < 4: d_even = 4
    if d_even % 2 == 1: d_even += 1
    d_candidates.append(d_even)
    
    # Choose d that minimizes q^d (which is d, since q>1)
    d = min(d_candidates)
    
    # If q==4, use GF(4)
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add = lambda x, y: x ^ y
        sub = lambda x, y: x ^ y
        mul = lambda x, y: table[x][y]
    else:
        add = lambda x, y: (x + y) % q
        sub = lambda x, y: (x - y) % q
        mul = lambda x, y: x * y % q
        
    # Rule for CD(d, q)
    # y[c] = y[a] * x[b] + x[c] (for points to lines)
    # The specific rule depends on d.
    # Standard rule:
    # For c=2..d:
    # If c is even: y[c] = y[c-1] * x[1] + x[c] ? No.
    # The rule in the seed:
    # rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    # for c in 5..d:
    #   r, c0 = (c-5)%4, c - (c-5)%4
    #   rule[c] = [(1, c0-2), (c0-1, 1), (c0, 1), (1, c0+1)][r]
    
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
    
    # BFS
    i = 0
    while i < len(order):
        vertex = order[i]
        i += 1
        if time.time() > deadline:
            break
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
                
    return len(order), edges
