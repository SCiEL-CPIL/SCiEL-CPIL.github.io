"""Program for constructing small k-regular graphs of given girth."""
import numpy as np
from collections import deque

# EVOLVE-BLOCK-START
def construct(k, g):
    """
    Construct a k-regular graph with girth at least g using the Lazebnik-Ustimenko-Woldar (LUW)
    construction. This gives an explicit family of cages.
    
    For k=3, g=16: we need girth >= 16.
    Using LUW: girth >= d+5 for odd d. So d+5 >= 16 => d >= 11.
    With q=3, d=11: gives a 3-regular bipartite graph with girth >= 16.
    """
    if k == 2:
        # Cycle graph
        n = max(g, 3)
        edges = [(i, (i + 1) % n) for i in range(n)]
        return n, edges
    
    if k == 1:
        # Single edge
        return 2, [(0, 1)]
    
    # Use Lazebnik-Ustimenko-Woldar construction
    # CD(d, q): q-regular bipartite graph, girth >= d+5 if d odd, >= d+4 if d even
    # We need girth >= g
    
    # Find appropriate d for given g
    # If d is odd: girth >= d + 5, so we need d + 5 >= g => d >= g - 5
    # If d is even: girth >= d + 4, so we need d + 4 >= g => d >= g - 4
    
    # We want to minimize the number of vertices, which grows as q^(d-1)
    # So we want the smallest d that satisfies the girth requirement
    
    candidates = []
    
    # Odd d
    d_odd = g - 5
    if d_odd < 3:
        d_odd = 3
    if d_odd % 2 == 0:
        d_odd += 1
    candidates.append(d_odd)
    
    # Even d
    d_even = g - 4
    if d_even < 2:
        d_even = 2
    if d_even % 2 == 1:
        d_even += 1
    candidates.append(d_even)
    
    # For q = k, pick the d that gives fewer vertices
    # Number of vertices ~ 2 * k^(d-1)
    # So smaller d is better
    
    d = min(candidates)
    q = k
    
    # Handle q=4 specially (GF(4))
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        def add(x, y): return x ^ y
        def sub(x, y): return x ^ y
        def mul(x, y): return table[x][y]
    else:
        def add(x, y): return (x + y) % q
        def sub(x, y): return (x - y) % q
        def mul(x, y): return (x * y) % q
    
    # Define the rule for LUW construction
    # rule[c] = (a, b) for c = 2, ..., d
    # The recurrence: for point p and line l, l[c] - p[c] = l[a] * p[b]
    
    rule = {}
    # Base cases
    rule[2] = (1, 1)
    rule[3] = (2, 1)
    rule[4] = (1, 2)
    
    for c in range(5, d + 1):
        # Pattern: r = (c-5) % 4
        r = (c - 5) % 4
        c0 = c - r
        if r == 0:
            rule[c] = (1, c0 - 2)
        elif r == 1:
            rule[c] = (c0 - 1, 1)
        elif r == 2:
            rule[c] = (c0, 1)
        else:
            rule[c] = (1, c0 + 1)
    
    # Precompute neighbor mappings for efficiency
    # For each side and each possible first coordinate, precompute the transformation
    # Actually, let's use a more efficient approach: represent vertices as tuples
    # and use a dictionary for the index
    
    def neighbours(vertex):
        side, x = vertex
        # x is a tuple of length d+1: x[0] is None, x[1]..x[d] are values
        for first in range(q):
            y = [0] * (d + 1)
            y[1] = first
            for c in range(2, d + 1):
                a, b = rule[c]
                if side == 0:
                    # point to line: y[c] = x[c] + y[a] * x[b]
                    y[c] = add(x[c], mul(y[a], x[b]))
                else:
                    # line to point: y[c] = x[c] - x[a] * y[b]
                    y[c] = sub(x[c], mul(x[a], y[b]))
            yield (1 - side, tuple(y))
    
    # BFS from the zero point
    start = (0, (0,) * (d + 1))
    index = {start: 0}
    order = [start]
    edges = []
    
    for i, vertex in enumerate(order):
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
    
    n = len(order)
    return n, edges
# EVOLVE-BLOCK-END
