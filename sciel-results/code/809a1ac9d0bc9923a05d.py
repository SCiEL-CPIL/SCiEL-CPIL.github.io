"""Construct a 5-regular graph with girth 9 using the Lazebnik-Ustimenko-Woldar construction."""
import time

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin before 120s limit

    # Use LUW construction: q-regular bipartite graph with girth >= d+5
    # For k=5, g=9: q=5, d=4 gives girth >= 9
    q = k
    # d is the number of coordinates (minus 2 for the first two)
    # Girth guarantee: d + 5 for odd d, d + 4 for even d
    # For g=9: need d+5 >= 9 => d >= 4 (odd d=5 gives girth>=10, d=4 gives girth>=8... wait)
    # Actually: for odd d, girth >= d+5; for even d, girth >= d+4
    # So for g=9: d=4 (even) gives girth >= 8, not enough
    # d=5 (odd) gives girth >= 10, which is >= 9
    # But the original code uses: d = max(3, g-5 if (g-5)%2 else g-4)
    # For g=9: g-5=4, 4%2==0, so d = max(3, 9-4) = max(3,5) = 5
    
    if (g - 5) % 2 == 0:
        d = max(3, g - 4)
    else:
        d = max(3, g - 5)
    
    # For g=9: d=5
    
    # Field operations for GF(5)
    def add(x, y):
        return (x + y) % q
    
    def sub(x, y):
        return (x - y) % q
    
    def mul(x, y):
        return (x * y) % q
    
    # Precompute the rule for each coordinate c from 2 to d
    # rule[c] = (a, b) such that l[c] - p[c] = l[a] * p[b]
    rule = {}
    rule[2] = (1, 1)
    rule[3] = (2, 1)
    rule[4] = (1, 2)
    for c in range(5, d + 1):
        r = (c - 5) % 4
        c0 = c - (c - 5) % 4
        candidates = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)]
        rule[c] = candidates[r]
    
    # Precompute multiplication tables for speed
    mul_table = [[mul(a, b) for b in range(q)] for a in range(q)]
    
    # Generate all vertices (points and lines) reachable from the zero point
    # Vertex representation: (side, coords) where side=0 is point, side=1 is line
    # coords is a tuple of length d+2: (None, c0, c1, ..., cd)
    
    start = (0, (None, 0, 0, 0, 0, 0))  # zero point with d=5, so 6 coords total (including None)
    
    index = {start: 0}
    order = [start]
    edges = []
    
    # Precompute neighbor relations using vectorized approach
    # For a point (0, p), neighbors are lines (1, l) where for each c in 2..d:
    #   l[c] - p[c] = l[a_c] * p[b_c]
    # For a line (1, l), neighbors are points (0, p) where for each c in 2..d:
    #   l[c] - p[c] = l[a_c] * p[b_c]  =>  p[c] = l[c] - l[a_c] * p[b_c]
    
    # Since this is bipartite and regular, each vertex has exactly q neighbors.
    # We'll do BFS to find the connected component.
    
    for vi in range(len(order)):
        if time.time() > deadline:
            break
        vertex = order[vi]
        side, x = vertex
        
        for first in range(q):
            y = [None, first] + [0] * d
            for c in range(2, d + 1):
                a, b = rule[c]
                if side == 0:
                    # Point to line: l[c] = p[c] + l[a] * p[b]
                    y[c] = add(x[c], mul_table[y[a]][x[b]])
                else:
                    # Line to point: p[c] = l[c] - l[a] * p[b]
                    # But p[b] is not yet known... this is a constraint satisfaction
                    # Actually in LUW, the relation is: l[c] - p[c] = l[a] * p[b]
                    # For line->point: p[c] = l[c] - l[a] * p[b]
                    # This creates a dependency chain
                    y[c] = sub(x[c], mul_table[x[a]][y[b]])
            
            other = (1 - side, tuple(y))
            
            if other not in index:
                index[other] = len(order)
                order.append(other)
            
            if side == 0:
                edges.append((index[vertex], index[other]))
    
    return len(order), edges
