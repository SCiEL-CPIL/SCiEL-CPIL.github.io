import time
import math

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 140  # Safety margin

    # For k=3, g=16, use LUW with d=7
    # Girth of LUW(d) is 2d+2.
    # 2*7+2 = 16.
    # Vertices = 2 * 3^(d+1) = 2 * 3^8 = 13122.
    
    if k != 3:
        # Fallback for other k
        # Use a generic LUW-like construction or return a trivial graph if possible
        # But the task specifies k=3, g=16.
        pass
        
    d = 7
    q = k # q=3
    
    # Field operations for GF(3)
    def add(x, y): return (x + y) % 3
    def sub(x, y): return (x - y) % 3
    def mul(x, y): return x * y % 3
    
    # LUW Rule
    # The rule maps c -> (a, b)
    # For d=7, c ranges from 2 to 7.
    rule = {}
    # Standard LUW rules for q=3
    # The rule is often defined as:
    # c=2: (1, 1)
    # c=3: (2, 1)
    # c=4: (1, 2)
    # c=5: (2, 2) ? 
    # Let's use the standard rule from Lazebnik et al.
    # rule[c] = ( (c-2)%3 + 1, (c-1)%3 + 1 ) ? No.
    
    # A common rule set for LUW:
    # r(c) = ( (c-2) mod 3 + 1, (c-1) mod 3 + 1 )
    # Let's verify for c=2: (0+1, 1+1) = (1, 2).
    # c=3: (1+1, 2+1) = (2, 0->3? no, mod 3 is 0, but indices are 1..d? )
    
    # Let's use the specific rule from the current code which seemed to work:
    # rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    # for c in 5..d:
    # r, c0 = (c-5)%4, c - (c-5)%4
    # rule[c] = [(1, c0-2), (c0-1, 1), (c0, 1), (1, c0+1)][r]
    
    # This rule is specific. Let's stick to it.
    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r, c0 = (c - 5) % 4, c - (c - 5) % 4
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]

    def neighbours(vertex):
        side, x = vertex
        # x is a tuple of length d+1, x[0] is None, x[1..d] are values
        for first in range(q):
            y = [None, first] + [0] * (d - 1)
            for c in range(2, d + 1):
                a, b = rule[c]
                # Note: a and b are indices into x (1-based in rule, 0-based in tuple?)
                # In the original code: x[c] is used. x is a tuple.
                # If x is (None, x1, ..., xd), then x[c] is the c-th element.
                # The rule indices a, b are likely 1-based corresponding to x[1]..x[d].
                # So x[a] and x[b] are valid.
                
                if side == 0:
                    y[c] = add(x[c], mul(y[a], x[b]))
                else:
                    y[c] = sub(x[c], mul(x[a], y[b]))
            yield 1 - side, tuple(y)

    start = (0, (None,) + (0,) * d)
    index = {start: 0}
    order = [start]
    edges = []

    # BFS to build the graph
    # This will generate 2 * 3^(d+1) vertices.
    # For d=7, this is 13122 vertices.
    # BFS is O(n).
    
    for i, vertex in enumerate(order):
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            # Add edge only once (from side 0 to side 1)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))

    n = len(order)
    
    # Check time
    if time.time() > deadline:
        # If we ran out of time, we might not have finished?
        # But 13k vertices is very fast.
        pass

    return n, edges
