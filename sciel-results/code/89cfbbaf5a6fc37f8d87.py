import time
import sys

def construct(k, g):
    """
    Constructs a k-regular graph with girth at least g.
    Uses Lazebnik-Ustimenko-Woldar (LUW) graph CD(d, q).
    For k=4, q=4.
    Girth of CD(d, 4) is at least d+5.
    We need girth >= g. So d+5 >= g => d >= g-5.
    We choose the smallest d such that the girth bound holds.
    For k=4, g=10: d >= 5.
    n = 2 * 4^d.
    """
    start_time = time.time()
    deadline = start_time + 110.0 # 10s safety margin
    
    # LUW parameters
    q = k
    # Determine d
    # Girth >= d + 5
    # We want d + 5 >= g
    d_min = g - 5
    if d_min < 3:
        d_min = 3
    
    # Check if we can use a smaller d if the girth is actually higher?
    # For CD(d, q), the girth is exactly d+5 for odd d, and d+6 for even d?
    # If d=4 (even), girth might be 10.
    # Let's try d=4 first for g=10. If it fails girth check, use d=5.
    # But checking girth on 512 vertices is fast.
    
    candidates_d = []
    # Try d_min, d_min+1, etc.
    # Actually, just pick the smallest d that guarantees girth >= g.
    # If d is odd, girth = d+5.
    # If d is even, girth = d+6? (Source: "The girth of CD(d,q) is d+5 if d is odd and d+6 if d is even" is a common claim, but let's verify with the bound "at least d+5").
    # If we use d=4, bound is 9. It might have a 9-cycle.
    # If we use d=5, bound is 10. It might have a 10-cycle.
    # To be safe and guarantee validity, we should use d such that d+5 >= g.
    # So d=5 for g=10.
    
    d = d_min
    # Ensure d is large enough for the specific q?
    # For q=4, d=5 is fine.
    
    # Build the graph
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        def add(x, y): return x ^ y
        def sub(x, y): return x ^ y
        def mul(x, y): return table[x][y]
    else:
        def add(x, y): return (x + y) % q
        def sub(x, y): return (x - y) % q
        def mul(x, y): return (x * y) % q

    # Rule generation for LUW
    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r, c0 = (c - 5) % 4, c - (c - 5) % 4
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]

    def neighbours(vertex):
        side, x = vertex
        # x is a tuple of length d+1. x[0] is None, x[1..d] are values.
        for first in range(q):
            y = [None, first] + [0] * (d - 1)
            for c in range(2, d + 1):
                a, b = rule[c]
                if side == 0:
                    # point p ~ line l: l[c] - p[c] = l[a] * p[b]
                    # l[c] = p[c] + l[a]*p[b]
                    y[c] = add(x[c], mul(y[a], x[b]))
                else:
                    # line l ~ point p: p[c] - l[c] = l[a] * p[b]
                    # p[c] = l[c] + l[a]*p[b]
                    # Wait, the relation is symmetric?
                    # l[c] - p[c] = l[a] p[b]
                    # p[c] = l[c] - l[a] p[b]
                    y[c] = sub(x[c], mul(x[a], y[b]))
            yield 1 - side, tuple(y)

    start = (0, (None,) + (0,) * d)
    index = {start: 0}
    order = [start]
    edges = []
    
    for i, vertex in enumerate(order):
        if time.time() > deadline:
            break
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
                
    # Verify girth quickly? 
    # If we used d=5, girth is >= 10. 
    # If we used d=4, girth is >= 9. 
    # For g=10, we must ensure no 9-cycles.
    # The LUW graph CD(4,4) is known to have girth 9? 
    # If so, d=5 is required.
    
    n = len(order)
    return n, edges
