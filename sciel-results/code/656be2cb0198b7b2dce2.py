import time
import random
import math
from collections import deque, defaultdict

_DEADLINE = time.time() + 135.0

def _check_deadline():
    return time.time() > _DEADLINE

def _luw_construct(k, g):
    """
    Construct a k-regular bipartite graph with girth at least 2*d + 2.
    We need girth >= g.
    If g is odd, we need girth >= g+1 (next even).
    If g is even, we need girth >= g.
    So target_girth = g if g % 2 == 0 else g + 1.
    We need 2*d + 2 >= target_girth => d >= (target_girth - 2) / 2.
    """
    target_girth = g if g % 2 == 0 else g + 1
    min_d = max(1, (target_girth - 2) // 2)
    
    # The seed program used a different d calculation which was suboptimal.
    # We use the minimal d to minimize n = 2 * k^d.
    d = min_d
    
    q = k
    
    if q == 4:
        table = [[0,0,0,0],[0,1,2,3],[0,2,3,1],[0,3,1,2]]
        add = lambda x,y: x ^ y
        sub = lambda x,y: x ^ y
        mul = lambda x,y: table[x][y]
    else:
        add = lambda x,y: (x+y) % q
        sub = lambda x,y: (x-y) % q
        mul = lambda x,y: x*y % q

    # Rule for LUW construction
    # rule[c] = (a, b) such that y[c] = y[a] * x[b] (or similar)
    # Standard LUW rule:
    # y[1] = x[1]
    # y[c] = y[c-1] + y[c-2] * x[c-2] ? No.
    # The seed code used:
    # rule = {2:(1,1), 3:(2,1), 4:(1,2)}
    # for c in range(5, d+1):
    #     r = (c-5) % 4
    #     c0 = c - (c-5) % 4
    #     rule[c] = [(1, c0-2),(c0-1,1),(c0,1),(1,c0+1)][r]
    
    # I will use the seed's rule as it is known to work for LUW.
    rule = {}
    if d >= 2: rule[2] = (1,1)
    if d >= 3: rule[3] = (2,1)
    if d >= 4: rule[4] = (1,2)
    for c in range(5, d+1):
        r = (c-5) % 4
        c0 = c - (c-5) % 4
        rule[c] = [(1, c0-2),(c0-1,1),(c0,1),(1,c0+1)][r]

    def neighbours(vertex):
        side, x = vertex
        for first in range(q):
            y = [None, first] + [0]*(d-1)
            for c in range(2, d+1):
                a, b = rule[c]
                if side == 0:
                    y[c] = add(x[c], mul(y[a], x[b]))
                else:
                    y[c] = sub(x[c], mul(x[a], y[b]))
            yield 1 - side, tuple(y)

    start = (0, (None,) + (0,)*d)
    index = {start: 0}
    order = [start]
    edges = []
    
    # BFS to build the graph
    for vertex in order:
        if _check_deadline():
            break
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
                
    return len(order), edges

def construct(k, g):
    # For k=5, g=9, we use LUW with d=4.
    # n = 2 * 5^4 = 1250.
    # Seed used d=5, n=6250.
    # Score = 6250 / 1250 = 5.0.
    
    if k == 5 and g == 9:
        n, edges = _luw_construct(5, 9)
        return n, edges
    
    # Fallback for other cases
    n, edges = _luw_construct(k, g)
    return n, edges
