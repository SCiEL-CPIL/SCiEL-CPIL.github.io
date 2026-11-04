"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct(k, g):
    """
    Constructs a 3-regular graph of girth at least g using the Lazebnik-Ustimenko-Woldar (LUW) family.
    The LUW graph CD(d, q) is q-regular. For q=3, the girth is approximately 2d+2.
    We select d such that the girth meets the requirement g.
    """
    import time
    start_time = time.time()
    deadline = 110.0  # Safety margin before 120s limit

    q = k
    # Heuristic: Girth of CD(d, q) is 2d + 2 for small q.
    # We need 2d + 2 >= g => d >= (g - 2) / 2.
    # We ensure d is at least 3 for the recurrence to be well-defined in this implementation.
    d = max(3, (g - 2) // 2)
    
    # If time runs out (unlikely for d=7, but good practice), we might want to try smaller d?
    # No, smaller d means smaller girth, which is invalid.
    # We must stick to the d that satisfies girth.

    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add, sub, mul = (lambda x, y: x ^ y), (lambda x, y: x ^ y), (lambda x, y: table[x][y])
    else:
        add, sub, mul = (lambda x, y: (x + y) % q), (lambda x, y: (x - y) % q), (lambda x, y: x * y % q)
        
    # Pre-compute rules for the LUW recurrence
    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r, c0 = (c - 5) % 4, c - (c - 5) % 4
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]

    def neighbours(vertex):
        side, x = vertex
        # x is a tuple of length d+1. x[0] is None, x[1] is first coord, etc.
        # Actually, the seed code uses:
        # y = [None, first] + [0] * (d - 1)
        # So y has length 1 + 1 + (d-1) = d+1.
        # Indices 0..d.
        for first in range(q):
            y = [None, first] + [0] * (d - 1)
            for c in range(2, d + 1):
                a, b = rule[c]
                # The indices a and b refer to positions in the vector.
                # In the seed code, x is a tuple.
                # y[c] is being calculated.
                # a and b are indices.
                # The seed code uses:
                # y[c] = add(x[c], mul(y[a], x[b])) if side == 0 else sub(x[c], mul(x[a], y[b]))
                # This implies x and y are indexed 0..d.
                # x[0] is None.
                # The rule indices a,b must be valid.
                # For c=2, rule is (1,1). y[2] = x[2] +/- y[1]x[1].
                # For c=3, rule is (2,1). y[3] = x[3] +/- y[2]x[1].
                # This looks like a standard linear recurrence.
                
                if side == 0:
                    y[c] = add(x[c], mul(y[a], x[b]))
                else:
                    y[c] = sub(x[c], mul(x[a], y[b]))
            yield 1 - side, tuple(y)

    start = (0, (None,) + (0,) * d)
    index, order, edges = {start: 0}, [start], []
    
    # BFS to build the graph
    for vertex in order:
        if time.time() > deadline:
            break
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            # Add edge only once (from side 0 to side 1)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
                
    return len(order), edges
# EVOLVE-BLOCK-END
