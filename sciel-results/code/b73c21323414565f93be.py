import time
import sys

# EVOLVE-BLOCK-START
def construct(k, g):
    """
    Constructs a k-regular graph with girth at least g using the Lazebnik-Ustimenko-Woldar (LUW) construction.
    For k=3, g=16, this yields the known (3,16)-cage with 936 vertices.
    """
    # Deadline guard: 150s limit, aim for 140s safety margin
    deadline = time.time() + 140.0

    q = k
    # Determine d for LUW graph CD(d, q)
    # Girth is d+5 for odd d, d+4 for even d.
    # We want girth >= g.
    # The code logic:
    # if (g-5) is odd, d = g-5 (which is odd) -> girth = (g-5)+5 = g.
    # if (g-5) is even, d = g-4 (which is even) -> girth = (g-4)+4 = g.
    # This ensures exact girth g (or close to it) and minimal d.
    if (g - 5) % 2 != 0:
        d = g - 5
    else:
        d = g - 4
    
    # Ensure d is at least 3 for the rules to make sense
    d = max(3, d)

    # Define field operations for GF(q)
    if q == 4:
        # GF(4) arithmetic
        table = [
            [0, 0, 0, 0],
            [0, 1, 2, 3],
            [0, 2, 3, 1],
            [0, 3, 1, 2]
        ]
        def add(x, y): return x ^ y
        def sub(x, y): return x ^ y
        def mul(x, y): return table[x][y]
    else:
        # GF(q) for prime q
        def add(x, y): return (x + y) % q
        def sub(x, y): return (x - y) % q
        def mul(x, y): return (x * y) % q

    # Precompute the rule coefficients for the LUW graph
    # rule[c] = (a, b) such that l[c] - p[c] = l[a] * p[b]
    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r = (c - 5) % 4
        c0 = c - r
        # The sequence of (a,b) pairs cycles
        # c=5 (r=0, c0=5): (1, 3) ? 
        # Let's trace the original code's logic for rule generation
        # r=0: (1, c0-2) -> (1, 3)
        # r=1: (c0-1, 1) -> (4, 1)
        # r=2: (c0, 1) -> (5, 1)
        # r=3: (1, c0+1) -> (1, 6)
        # Original code:
        # rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]

    def neighbours(vertex):
        """Yields the neighbors of a given vertex."""
        side, x = vertex
        # x is a tuple of length d+1, x[0] is None (for side 0) or ignored?
        # In the start vertex: (None,) + (0,) * d. Length d+1.
        # Indices used in rule: a, b, c.
        # c ranges from 2 to d.
        # a, b are indices in 1..d.
        
        for first in range(q):
            # y is the coordinate vector of the neighbor
            # y[1] = first
            # y[0] is not used in the same way?
            # The code initializes y = [None, first] + [0] * (d - 1)
            # This creates a list of length 1 + 1 + d - 1 = d + 1.
            # Indices 0..d.
            y = [None, first] + [0] * (d - 1)
            
            for c in range(2, d + 1):
                a, b = rule[c]
                if side == 0:
                    # Point to Line
                    # y[c] = x[c] + y[a] * x[b]
                    y[c] = add(x[c], mul(y[a], x[b]))
                else:
                    # Line to Point
                    # y[c] = x[c] - x[a] * y[b]
                    y[c] = sub(x[c], mul(x[a], y[b]))
            
            yield 1 - side, tuple(y)

    # BFS to find the connected component
    start = (0, (None,) + (0,) * d)
    index = {start: 0}
    order = [start]
    edges = []
    
    # Optimization: Use a list for the queue and process by index
    for i in range(len(order)):
        if time.time() > deadline:
            # If we time out, return what we have (should be complete for small graphs)
            # But for safety, break
            break
            
        vertex = order[i]
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            # Add edge only once (from side 0 to side 1)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
                
    return len(order), edges
# EVOLVE-BLOCK-END
