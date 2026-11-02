"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct(k, g):
    # Lazebnik-Ustimenko-Woldar graph CD(d, q) with q = k.
    # The girth of LUW(d, q) is 2d + 2 for d >= 1.
    # We need girth >= g.
    # So we need 2d + 2 >= g => d >= (g - 2) / 2.
    # d must be an integer.
    
    # Calculate minimal d
    min_d = max(1, (g - 2 + 1) // 2)  # Ceiling of (g-2)/2
    
    # For very small g, d=1 might be too small (girth 4).
    # If g <= 4, d=1 is fine.
    # If g=5, d=2 (girth 6).
    # If g=9, d=4 (girth 10).
    
    d = min_d
    q = k
    
    # GF(q) operations
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add = lambda x, y: x ^ y
        sub = lambda x, y: x ^ y
        mul = lambda x, y: table[x][y]
    else:
        # Assume q is prime for simplicity in this improvement, 
        # or handle general GF(q) if needed. 
        # For k=5, q=5 is prime.
        add = lambda x, y: (x + y) % q
        sub = lambda x, y: (x - y) % q
        mul = lambda x, y: (x * y) % q

    # Rule for LUW construction
    # The rule defines how coordinates are computed.
    # Standard LUW rules:
    # For c from 2 to d:
    # y[c] depends on y[c-1], y[c-2], etc.
    # The specific rule in the seed was:
    # rule[c] = (a, b) such that l[c] - p[c] = l[a] * p[b]
    # The seed used a specific sequence of (a,b).
    # We will replicate the seed's rule logic but with the corrected d.
    
    rule = {}
    # The seed's rule logic:
    # rule[2] = (1, 1)
    # rule[3] = (2, 1)
    # rule[4] = (1, 2)
    # For c >= 5:
    # r = (c - 5) % 4
    # c0 = c - r
    # rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]
    
    if d >= 2: rule[2] = (1, 1)
    if d >= 3: rule[3] = (2, 1)
    if d >= 4: rule[4] = (1, 2)
    for c in range(5, d + 1):
        r = (c - 5) % 4
        c0 = c - r
        # The indices in the tuple y are 0..d.
        # y[0] is None, y[1] is 'first', y[2..d] are computed.
        # The rule uses indices a, b which refer to positions in the vector.
        # In the seed:
        # y[c] = add(x[c], mul(y[a], x[b]))
        # x is the point, y is the line.
        # Indices 1..d are valid.
        # The seed's rule values:
        # c=5: r=0, c0=5. rule=(1, 3). 
        # c=6: r=1, c0=6. rule=(5, 1).
        # c=7: r=2, c0=7. rule=(7, 1).
        # c=8: r=3, c0=8. rule=(1, 9).
        
        # Wait, the indices in the rule must be within 1..d.
        # If c=8, c0=8, rule=(1, 9). But d=4 in our case, so we don't reach c=8.
        # For d=4, we only need rules up to 4.
        
        # Let's re-verify the rule for c=5 if d>=5.
        # The seed code:
        # rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]
        # For c=5, r=0, c0=5. rule = (1, 3).
        # For c=6, r=1, c0=6. rule = (5, 1).
        # For c=7, r=2, c0=7. rule = (7, 1).
        # For c=8, r=3, c0=8. rule = (1, 9).
        
        # We just copy this logic.
        if r == 0:
            rule[c] = (1, c0 - 2)
        elif r == 1:
            rule[c] = (c0 - 1, 1)
        elif r == 2:
            rule[c] = (c0, 1)
        else:
            rule[c] = (1, c0 + 1)

    def neighbours(vertex):
        side, x = vertex
        # x is a tuple of length d+1: (None, x1, x2, ..., xd)
        for first in range(q):
            # y is the line vector
            y = [None, first] + [0] * (d - 1)
            for c in range(2, d + 1):
                a, b = rule[c]
                if side == 0:
                    # Point to Line
                    # l[c] - p[c] = l[a] * p[b]
                    # y[c] = x[c] + y[a] * x[b]
                    y[c] = add(x[c], mul(y[a], x[b]))
                else:
                    # Line to Point
                    # p[c] - l[c] = l[a] * p[b] ? 
                    # The seed: y[c] = sub(x[c], mul(x[a], y[b]))
                    # Here x is the line, y is the point.
                    # So point[c] = line[c] - line[a] * point[b]
                    y[c] = sub(x[c], mul(x[a], y[b]))
            yield 1 - side, tuple(y)

    # Start from zero point
    start = (0, (None,) + (0,) * d)
    index = {start: 0}
    order = [start]
    edges = []
    
    # BFS to find the connected component
    for vertex in order:
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
                
    return len(order), edges
# EVOLVE-BLOCK-END
