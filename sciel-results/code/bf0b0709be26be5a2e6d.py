"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct(k, g):
    """
    Construct a k-regular graph with girth at least g on as few vertices as possible.
    Uses the Lazebnik-Ustimenko-Woldar construction CD(d, q) with q = k.
    """
    start_time = time.time()
    deadline = start_time + 115  # safety margin

    q = k
    # LUW graph CD(d, q) has girth at least d + 5 (for odd d) or d + 4 (for even d).
    # We need girth >= g.
    # For d even: girth >= d + 4, so d >= g - 4. Smallest even d >= g-4.
    # For d odd:  girth >= d + 5, so d >= g - 5. Smallest odd d >= g-5.
    # We want to minimize d (and thus n = 2*q^d), so pick the smaller of the two options.
    
    # Option 1: smallest even d such that d + 4 >= g -> d >= g - 4
    d_even_candidate = g - 4
    if d_even_candidate < 2:
        d_even_candidate = 2
    if d_even_candidate % 2 != 0:
        d_even_candidate += 1
    
    # Option 2: smallest odd d such that d + 5 >= g -> d >= g - 5
    d_odd_candidate = g - 5
    if d_odd_candidate < 1:
        d_odd_candidate = 1
    if d_odd_candidate % 2 != 1:
        d_odd_candidate += 1
    
    # Pick the smaller d
    if d_even_candidate <= d_odd_candidate:
        d = d_even_candidate
    else:
        d = d_odd_candidate
    
    # Ensure d is at least 2 for the construction to make sense
    d = max(2, d)

    # Define field operations
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add = lambda x, y: x ^ y
        sub = lambda x, y: x ^ y
        mul = lambda x, y: table[x][y]
    else:
        add = lambda x, y: (x + y) % q
        sub = lambda x, y: (x - y) % q
        mul = lambda x, y: x * y % q

    # Define the recurrence rule for the LUW construction
    # The rule is a sequence of pairs (a, b) for c = 2, 3, ..., d
    # For the standard LUW construction:
    # rule[c] = (a, b) means y[c] = x[c] + y[a] * x[b] (for points to lines)
    # The specific pattern depends on d mod 4 and the position c.
    
    # Standard LUW rule pattern (from literature):
    # For c = 2..d, the pair (a_c, b_c) follows a periodic pattern.
    # One common pattern for CD(d,q):
    # a_c = 1, b_c = c-2 for c odd? No, let's use the known formula.
    
    # Actually, the seed code had a specific rule. Let's use a known valid rule.
    # A standard rule for LUW graphs:
    # For i = 2..d:
    #   if i is even: a_i = 1, b_i = i-2
    #   if i is odd:  a_i = i-1, b_i = 1
    # Wait, that's not quite right. Let's use the rule from the seed but fix d.
    
    # The seed's rule:
    # rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    # for c in 5..d: r, c0 = (c-5)%4, c-(c-5)%4
    #   rule[c] = [(1, c0-2), (c0-1, 1), (c0, 1), (1, c0+1)][r]
    
    rule = {}
    rule[2] = (1, 1)
    rule[3] = (2, 1)
    rule[4] = (1, 2)
    for c in range(5, d + 1):
        r, c0 = (c - 5) % 4, c - (c - 5) % 4
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]

    def neighbours(vertex):
        side, x = vertex
        # x is a tuple of length d+1, with x[0] = None (unused)
        for first in range(q):
            y = [None, first] + [0] * (d - 1)
            for c in range(2, d + 1):
                a, b = rule[c]
                if side == 0:
                    y[c] = add(x[c], mul(y[a], x[b]))
                else:
                    y[c] = sub(x[c], mul(x[a], y[b]))
            yield 1 - side, tuple(y)

    start_vertex = (0, (None,) + (0,) * d)
    
    # BFS to find connected component
    index = {start_vertex: 0}
    order = [start_vertex]
    edges = []
    
    for i, vertex in enumerate(order):
        if time.time() > deadline:
            break
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            # Only add edge from point side to avoid duplicates
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
    
    return len(order), edges
# EVOLVE-BLOCK-END
