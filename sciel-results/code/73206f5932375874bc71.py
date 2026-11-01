import time
import itertools
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    if k == 4 and g == 10:
        # Try to find a circulant graph
        n, edges = _find_circulant_cage(4, 10, deadline)
        if n is not None:
            return n, edges
        
    # Fallback to LUW
    return _luw_construct(k, g)

def _luw_construct(k, g):
    # Original seed code logic
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

def _find_circulant_cage(k, g, deadline):
    # Search for circulant graph C_n(S) with girth >= g
    # n must be >= Moore bound
    moore_bound = 1 + k * sum((k-1)**i for i in range(g//2))
    # For k=4, g=10, moore_bound = 161
    
    # We search n from moore_bound to some limit
    # If we find one, return it
    # We prioritize small n
    
    # To speed up, we can check specific known candidates or just scan
    # Scanning all n is expensive.
    # Let's try a range.
    
    # If time is short, just return None
    if time.time() > deadline - 10:
        return None, None
        
    for n in range(moore_bound, 400):
        if time.time() > deadline - 5:
            return None, None
            
        # S is a set of k distinct steps from 1 to n//2
        # For k=4, we choose 4 steps
        # We can restrict to small steps to avoid short cycles?
        # Actually, small steps cause short cycles.
        # We should sample or iterate smartly.
        
        # For now, let's just iterate combinations of 4 steps from 1..n//2
        # This is too many.
        
        # Heuristic: Try steps that are "far apart"
        # Or just try a few random samples?
        # Deterministic is better for reproducibility.
        
        # Let's try a specific known construction if available?
        # No, let's just do a limited search.
        
        # Optimization: Only check if n is even?
        # Circulant graphs with odd n are also possible.
        
        pass
    return None, None
