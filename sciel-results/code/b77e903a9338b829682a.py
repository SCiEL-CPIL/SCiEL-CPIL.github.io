import time
import sys

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # Construct the LUW graph
    n, edges = _construct_luw(k, g)
    
    # The LUW graph is a valid solution.
    # For k=5, g=9, n=1250.
    # We return this as it is the most reliable construction available.
    # Random search for smaller n is not feasible within the time limit in Python.
    
    return n, edges

def _construct_luw(k, g):
    # Lazebnik-Ustimenko-Woldar construction
    # This constructs a k-regular graph of girth at least g.
    # For k=5, g=9, it produces a graph with 1250 vertices.
    
    q = k
    # Determine d based on g
    # The standard LUW construction uses d = (g-1)/2 for odd g?
    # The code in the prompt used:
    # d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    # For g=9, d=5.
    # Let's stick to the logic that produced the known 1250 result.
    
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    
    # Define operations
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add = lambda x, y: x ^ y
        sub = lambda x, y: x ^ y
        mul = lambda x, y: table[x][y]
    else:
        add = lambda x, y: (x + y) % q
        sub = lambda x, y: (x - y) % q
        mul = lambda x, y: x * y % q
        
    # Define the rule for the recurrence
    # rule[c] = (a, b) such that y[c] = op(y[a], x[b])
    rule = {}
    rule[2] = (1, 1)
    rule[3] = (2, 1)
    rule[4] = (1, 2)
    for c in range(5, d + 1):
        r = (c - 5) % 4
        c0 = c - r
        # The sequence of coefficients
        # Based on the prompt's code:
        # [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]
        # Note: c0 - 2 can be negative or 0, which is fine as indices?
        # In the prompt, y[0] is None.
        # If c0=5, r=0 -> (1, 3).
        # If c0=5, r=1 -> (4, 1).
        # If c0=5, r=2 -> (5, 1) -> y[5] depends on y[5]? No, y[c] is being computed.
        # Wait, the prompt code:
        # y[c] = add(x[c], mul(y[a], x[b]))
        # If a=c, it's self-referential?
        # Let's check the prompt's rule generation carefully.
        # r = (c-5)%4.
        # If c=5, r=0, c0=5. rule[5] = (1, 3).
        # If c=6, r=1, c0=5. rule[6] = (4, 1).
        # If c=7, r=2, c0=5. rule[7] = (5, 1). y[7] depends on y[5].
        # If c=8, r=3, c0=5. rule[8] = (1, 6).
        # This looks like a linear recurrence.
        
        options = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)]
        rule[c] = options[r]

    def neighbours(vertex):
        side, x = vertex
        for first in range(q):
            # y is a list of length d+1
            # y[0] is unused (None)
            # y[1] = first
            # y[2..d] are computed
            y = [None] * (d + 1)
            y[1] = first
            for c in range(2, d + 1):
                a, b = rule[c]
                if side == 0:
                    # Forward
                    val = mul(y[a], x[b])
                    y[c] = add(x[c], val)
                else:
                    # Backward
                    val = mul(x[a], y[b])
                    y[c] = sub(x[c], val)
            yield 1 - side, tuple(y)

    start = (0, (None,) + (0,) * d)
    index = {start: 0}
    order = [start]
    edges = []
    
    # BFS to generate all vertices
    for i, vertex in enumerate(order):
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            # Add edge if not already added
            # To avoid duplicates, we only add if vertex index < other index?
            # Or just collect all and deduplicate?
            # The LUW graph is simple, so we can just add edges from side 0 vertices
            # to ensure each edge is counted once.
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
                
    return len(order), edges
