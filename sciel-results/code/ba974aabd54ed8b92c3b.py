"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct(k, g):
    """
    Construct a k-regular graph with girth >= g on as few vertices as possible.
    Uses the Lazebnik-Ustimenko-Woldar (LUW) graph construction.
    """
    start_time = time.time()
    deadline = start_time + 120.0  # 120s limit, safety margin included in logic if needed, but we just run fast.
    
    # LUW parameters
    q = k
    # For girth g, we need d such that girth is at least g.
    # Girth of CD(d,q) is d+5 for odd d, d+6 for even d?
    # Seed used: d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    # For g=10: g-5=5 (odd) -> d=5. Girth = 5+5=10. Correct.
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    
    # Field operations
    if q == 4:
        # GF(4)
        table_add = [
            [0, 1, 2, 3],
            [1, 0, 3, 2],
            [2, 3, 0, 1],
            [3, 2, 1, 0]
        ]
        table_mul = [
            [0, 0, 0, 0],
            [0, 1, 2, 3],
            [0, 2, 3, 1],
            [0, 3, 1, 2]
        ]
        def add(x, y): return table_add[x][y]
        def sub(x, y): return table_add[x][y] # In char 2, -x = x
        def mul(x, y): return table_mul[x][y]
    else:
        # GF(q) for prime q
        def add(x, y): return (x + y) % q
        def sub(x, y): return (x - y) % q
        def mul(x, y): return (x * y) % q

    # Rule definition
    # rule[c] = (a, b) such that y[c] = y[a] * x[b] + x[c] (for points)
    # The seed uses a specific rule mapping.
    rule = {}
    rule[2] = (1, 1)
    rule[3] = (2, 1)
    rule[4] = (1, 2)
    for c in range(5, d + 1):
        r = (c - 5) % 4
        c0 = c - (c - 5) % 4
        # The seed's rule selection logic
        options = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)]
        rule[c] = options[r]

    def neighbours(vertex):
        side, x = vertex
        # x is a tuple of length d+1, with x[0]=None
        # Actually, seed uses: start = (0, (None,) + (0,) * d)
        # So x has length d+1.
        # Loop c from 2 to d.
        for first in range(q):
            y = [None, first] + [0] * (d - 1)
            # y[c] depends on x and previous y
            for c in range(2, d + 1):
                a, b = rule[c]
                if side == 0:
                    # Point p ~ Line l: l[c] - p[c] = l[a] * p[b]
                    # y[c] = x[c] + y[a] * x[b]
                    y[c] = add(x[c], mul(y[a], x[b]))
                else:
                    # Line l ~ Point p: l[c] - p[c] = l[a] * p[b]
                    # x[c] - y[c] = x[a] * y[b] => y[c] = x[c] - x[a] * y[b]
                    y[c] = sub(x[c], mul(x[a], y[b]))
            yield 1 - side, tuple(y)

    # BFS to find connected component
    start_vertex = (0, (None,) + (0,) * d)
    index = {start_vertex: 0}
    order = [start_vertex]
    edges = []
    
    head = 0
    while head < len(order):
        if time.time() > deadline - 5.0:
            # Break out if taking too long, though BFS is O(n)
            break
        v = order[head]
        head += 1
        for u in neighbours(v):
            if u not in index:
                index[u] = len(order)
                order.append(u)
            # Add edge only from one side to avoid duplicates
            if v[0] == 0:
                edges.append((index[v], index[u]))

    return len(order), edges
# EVOLVE-BLOCK-END
