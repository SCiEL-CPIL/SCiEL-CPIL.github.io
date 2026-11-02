"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
# EVOLVE-BLOCK-START
_DEADLINE = None

def _time_left():
    if _DEADLINE is None:
        return 1e9
    return _DEADLINE - time.monotonic()

def construct(k, g):
    global _DEADLINE
    _DEADLINE = time.monotonic() + 115.0
    q = k
    # For LUW CD(d,q): girth >= d+5 if d odd, d+4 if d even
    # We need girth >= g, so:
    #   if d odd: d >= g-5
    #   if d even: d >= g-4
    # Try the smallest valid d first
    d_candidates = []
    for d in range(3, 50):
        min_girth = d + 5 if d % 2 == 1 else d + 4
        if min_girth >= g:
            d_candidates.append(d)
            break
    d = d_candidates[0] if d_candidates else max(3, g - 5)

    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add_fn = lambda x, y: x ^ y
        sub_fn = lambda x, y: x ^ y
        mul_fn = lambda x, y: table[x][y]
    else:
        add_fn = lambda x, y: (x + y) % q
        sub_fn = lambda x, y: (x - y) % q
        mul_fn = lambda x, y: x * y % q

    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r, c0 = (c - 5) % 4, c - (c - 5) % 4
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]

    # Precompute rule lists for speed
    rule_list = [None, None] + [rule[c] for c in range(2, d + 1)]

    def neighbours_flat(side, x_tuple):
        x = list(x_tuple)
        x[0] = None
        for first in range(q):
            y = [None, first] + [0] * (d - 1)
            for c in range(2, d + 1):
                a, b = rule_list[c]
                if side == 0:
                    y[c] = add_fn(x[c], mul_fn(y[a], x[b]))
                else:
                    y[c] = sub_fn(x[c], mul_fn(x[a], y[b]))
            yield 1 - side, tuple(y)

    start = (0, (None,) + (0,) * d)
    index = {start: 0}
    order = [start]
    edges = []
    i = 0
    while i < len(order):
        if _time_left() < 3.0:
            break
        vertex = order[i]
        side, x = vertex
        for other in neighbours_flat(side, x):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if side == 0:
                edges.append((index[vertex], index[other]))
        i += 1

    n = len(order)
    return n, edges
# EVOLVE-BLOCK-END
