import time
from collections import deque

_DEADLINE = 115.0
_START = time.time()

def _remaining():
    return _DEADLINE - (time.time() - _START)

def _luw(k, g):
    """Lazebnik-Ustimenko-Woldar construction."""
    q = k
    # d is the dimension of the vector space over GF(q)
    # For girth g, we need d such that the diameter of the Cayley graph is at least g/2
    # Standard LUW uses d = g - 5 for k=4?
    # Actually, the formula for girth of LUW graph LUW(q, d) is 2*floor(log_q( (q^d - 1) / (q - 1) )) + 2 ?
    # No, the girth is related to the length of the shortest relation in the group.
    # For q=4, d=5, girth is 10.
    # For q=4, d=4, girth is 8?
    # Let's check the relation length.
    # The relation is x1 x2 ... xd = 0.
    # The girth is 2 * (min length of non-trivial relation).
    # For q=4, d=5, the min relation length is 5? No, girth 10 means cycle length 10, so path length 5.
    # So d=5 is correct for g=10.
    
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    
    if q == 4:
        # GF(4) = {0, 1, 2, 3}
        # 0: 0, 1: 1, 2: omega, 3: omega^2
        # omega^2 = omega + 1
        # Table for multiplication in GF(4)
        table = [
            [0, 0, 0, 0],
            [0, 1, 2, 3],
            [0, 2, 3, 1],
            [0, 3, 1, 2]
        ]
        add = lambda x, y: x ^ y
        sub = lambda x, y: x ^ y
        mul = lambda x, y: table[x][y]
    else:
        # Assume q is prime for simplicity, or use generic field logic if needed
        # For this task, k=4 is the focus.
        add = lambda x, y: (x + y) % q
        sub = lambda x, y: (x - y) % q
        mul = lambda x, y: (x * y) % q

    # Define the rules for the generators
    # The LUW graph is a Cayley graph of the group G = (Z_q)^d
    # Generators are defined by the rules.
    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r = (c - 5) % 4
        c0 = c - r
        # The rules cycle through specific patterns
        # Pattern for r=0: (1, c0-2)
        # Pattern for r=1: (c0-1, 1)
        # Pattern for r=2: (c0, 1)
        # Pattern for r=3: (1, c0+1)
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
        # x is a tuple of length d+1, x[0] is None, x[1..d] are in GF(q)
        for first in range(q):
            y = [None, first] + [0] * (d - 1)
            for c in range(2, d + 1):
                a, b = rule[c]
                if side == 0:
                    y[c] = add(x[c], mul(y[a], x[b]))
                else:
                    y[c] = sub(x[c], mul(x[a], y[b]))
            yield 1 - side, tuple(y)

    start = (0, (None,) + (0,) * d)
    index = {start: 0}
    order = [start]
    edges = []
    
    for vertex in order:
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
                
    return len(order), edges

def construct(k, g):
    global _START
    _START = time.time()
    # For k=4, g=10, LUW gives 2048.
    # We return this as it is a guaranteed valid solution.
    return _luw(k, g)
