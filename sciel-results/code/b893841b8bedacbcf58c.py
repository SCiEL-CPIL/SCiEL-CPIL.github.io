import time
import sys

# EVOLVE-BLOCK-START
def construct(k, g):
    """
    Construct a k-regular graph of girth at least g using the Lazebnik-Ustimenko-Woldar (LUW) construction.
    For k=4 and g=10, this yields the 320-vertex cage.
    """
    start_time = time.time()
    deadline = start_time + 115.0  # 5s safety margin

    # LUW parameters
    # q must be k for k-regularity
    q = k
    # d determines girth: girth >= d + 5
    # We need girth >= g.
    # So d + 5 >= g => d >= g - 5.
    # The code uses a specific parity adjustment for d.
    # Original: d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    # Let's stick to the original logic to ensure correctness.
    
    if (g - 5) % 2 == 1:
        d = g - 5
    else:
        d = g - 4
        
    if d < 3:
        d = 3

    # Define field operations for GF(q)
    if q == 4:
        # GF(4) using XOR for addition and a table for multiplication
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
        # Prime field GF(q)
        def add(x, y): return (x + y) % q
        def sub(x, y): return (x - y) % q
        def mul(x, y): return (x * y) % q

    # Determine the rule coefficients for the LUW construction
    # rule[c] = (a, b) such that l[c] - p[c] = l[a] * p[b]
    rule = {}
    # Base cases for small d
    if d >= 2: rule[2] = (1, 1)
    if d >= 3: rule[3] = (2, 1)
    if d >= 4: rule[4] = (1, 2)
    
    for c in range(5, d + 1):
        r = (c - 5) % 4
        c0 = c - r
        # The pattern of indices (a, b) repeats every 4
        # r=0: (1, c0-2)
        # r=1: (c0-1, 1)
        # r=2: (c0, 1)
        # r=3: (1, c0+1)
        if r == 0:
            rule[c] = (1, c0 - 2)
        elif r == 1:
            rule[c] = (c0 - 1, 1)
        elif r == 2:
            rule[c] = (c0, 1)
        else:
            rule[c] = (1, c0 + 1)

    # BFS to construct the graph
    # Vertices are represented as (side, x_tuple)
    # side 0: points, side 1: lines
    # x_tuple has length d+1 (indices 0..d), but index 0 is unused/None in the original logic?
    # Original: y = [None, first] + [0]*(d-1). Length 1 + 1 + d-1 = d+1.
    # Indices used: 1..d.
    
    start_vertex = (0, (0,) * (d + 1)) # Using 0 for None position to make it hashable and simple
    # Wait, original used None. Tuples with None are hashable.
    # Let's use integers 0..q-1 for all positions to be safe and fast.
    # The "first" coordinate is the variable part.
    
    index_map = {start_vertex: 0}
    order = [start_vertex]
    edges = []
    
    # Pre-compute neighbours for speed?
    # The neighbour generation is O(d) per neighbour.
    
    def get_neighbours(vertex):
        side, x = vertex
        # x is a tuple of length d+1
        # We iterate over the 'first' coordinate (index 1 in the tuple, which corresponds to variable 1)
        # Wait, the original code:
        # y = [None, first] + [0]*(d-1)
        # This sets y[0]=None, y[1]=first, y[2..d]=0.
        # Then it updates y[c] for c=2..d.
        # So the "free" variable is at index 1.
        
        for first in range(q):
            # Initialize y
            # y[0] is not used in the rule (rules use indices 1..d)
            # But it is part of the vertex identity.
            # In the original, side 0 vertices have x[0]=None?
            # No, start = (0, (None,) + (0,)*d).
            # So x[0] is None for all points?
            # And for lines?
            # The neighbour of a point (0, p) is a line (1, l).
            # The neighbour of a line (1, l) is a point (0, p).
            # The structure is symmetric.
            
            # Let's just mimic the original logic exactly to avoid bugs.
            y = [0] * (d + 1) # Using 0 for the "None" slot is fine if we never check it
            y[1] = first
            # y[2..d] are initially 0, but will be overwritten
            
            for c in range(2, d + 1):
                a, b = rule[c]
                if side == 0:
                    # Point p ~ Line l: l[c] - p[c] = l[a] * p[b]
                    # We are computing l (y) given p (x).
                    # y[c] = p[c] + l[a] * p[b]
                    # But l[a] is y[a].
                    # This is a recursive definition?
                    # No, the original code:
                    # y[c] = add(x[c], mul(y[a], x[b])) if side == 0
                    # This implies y[a] is already known when computing y[c].
                    # Since a < c (indices in rule are smaller than c?
                    # rule[2]=(1,1) -> a=1. c=2. 1<2.
                    # rule[3]=(2,1) -> a=2. c=3. 2<3.
                    # rule[4]=(1,2) -> a=1. c=4. 1<4.
                    # rule[5]=(1,3) -> a=1. c=5. 1<5.
                    # rule[6]=(5,1) -> a=5. c=6. 5<6.
                    # rule[7]=(6,1) -> a=6. c=7. 6<7.
                    # rule[8]=(1,5) -> a=1. c=8. 1<8.
                    # Yes, a < c always holds for this rule set.
                    # So y[a] is computed in a previous iteration.
                    y[c] = add(x[c], mul(y[a], x[b]))
                else:
                    # Line l ~ Point p: p[c] - l[c] = p[a] * l[b]
                    # We are computing p (y) given l (x).
                    # y[c] = l[c] - p[a] * l[b]
                    # p[a] is y[a].
                    y[c] = sub(x[c], mul(x[a], y[b]))
            
            neighbour = (1 - side, tuple(y))
            yield neighbour

    # BFS
    for i, vertex in enumerate(order):
        if time.time() > deadline:
            # Return whatever we have so far? 
            # The graph must be regular. If we cut off BFS, it won't be regular.
            # So we should not return a partial graph.
            # If we run out of time, we return the seed result (320) or an empty graph?
            # The task says "invalid result... scores 0".
            # Better to finish if possible. The LUW graph is small (320 nodes).
            # It should finish in < 1s.
            pass
        
        for other in get_neighbours(vertex):
            if other not in index_map:
                index_map[other] = len(order)
                order.append(other)
            # Add edge only once (from point to line)
            if vertex[0] == 0:
                edges.append((index_map[vertex], index_map[other]))
        
        # Check time periodically
        if i % 10 == 0 and time.time() > deadline:
             # If we really time out, we can't return a valid partial graph.
             # We'll just let it run, as 320 nodes is fast.
             pass

    return len(order), edges
# EVOLVE-BLOCK-END
