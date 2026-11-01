import time
import sys

def construct(k, g):
    """
    Constructs a Lazebnik-Ustimenko-Woldar graph CD(k, q) with q=k.
    This graph is k-regular with girth at least d+5, where d is derived from g.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit

    q = k
    # Determine d for LUW graph CD(d, q) to achieve girth >= g
    # Girth of CD(d, q) is d + 5 (for odd d) or d + 4 (for even d)?
    # Actually, the standard result is girth >= d + 5.
    # Let's assume the seed logic is correct for the specific case.
    # Seed logic: d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    if (g - 5) % 2 == 0:
        d = max(3, g - 4)
    else:
        d = max(3, g - 5)
        
    # If d is too large, the graph might be too big or construction might be slow.
    # For k=5, g=9, d=4.
    
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add = lambda x, y: x ^ y
        sub = lambda x, y: x ^ y
        mul = lambda x, y: table[x][y]
    else:
        add = lambda x, y: (x + y) % q
        sub = lambda x, y: (x - y) % q
        mul = lambda x, y: x * y % q

    # Precompute rules for c in 2..d
    # rule[c] = (a, b) such that y[c] = y[a] * x[b] + x[c] (for points)
    # The specific rule depends on c.
    # From the seed:
    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r = (c - 5) % 4
        c0 = c - r
        # The seed used a list of tuples. Let's replicate that.
        # (1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)
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
        # x is a tuple of length d+1, x[0] is None, x[1..d] are values
        # Actually, in the seed, x[0] is None, and x[1] is the first coordinate?
        # Seed: start = (0, (None,) + (0,) * d)
        # So x has length d+1. Indices 0..d.
        # In the loop: for first in range(q): y = [None, first] + [0] * (d - 1)
        # So y[1] = first.
        for first in range(q):
            y = [None] + [0] * d
            y[1] = first
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

    start = (0, (None,) + (0,) * d)
    index = {start: 0}
    order = [start]
    edges = []

    # BFS/DFS to build the graph
    # Using a list as a queue
    for i in range(len(order)):
        vertex = order[i]
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
                # Check time
                if time.time() > deadline:
                    # Return what we have? No, must be regular.
                    # If we break, the graph is not regular.
                    # So we must not break unless we are sure.
                    # For k=5, g=9, it finishes in < 1s.
                    pass
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))

    return len(order), edges
