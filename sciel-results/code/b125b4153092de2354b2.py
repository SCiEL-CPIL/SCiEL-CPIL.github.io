import time
import itertools
from collections import deque

def construct(k, g):
    """
    Construct a k-regular graph with girth at least g on as few vertices as possible.
    Returns (n, edges) where edges is a list of (u, v) on vertices 0..n-1.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit

    # For the specific case k=4, g=10, try to find a Cayley graph on Z_16 x Z_20 (320 vertices)
    if k == 4 and g == 10:
        result = _try_cayley_320(deadline)
        if result is not None:
            return result

    # Fallback: LUW Construction
    return _luw_construct(k, g)

def _try_cayley_320(deadline):
    """
    Try to find a 4-regular Cayley graph on Z_16 x Z_20 with girth >= 10.
    Generators: a, -a, b, -b.
    We fix a = (1, 0).
    We search for b = (x, y).
    """
    m, n = 16, 20
    total = m * n

    def check_girth_fast(x, y):
        """Check if Cayley graph with generators {(1,0),(-1,0),(x,y),(-x,-y)} has girth >= 10."""
        g0 = n
        g1 = -n
        g2 = x * n + y
        g3 = (-x * n - y) % total
        gens = (g0, g1, g2, g3)

        depth = [-1] * total
        parent = [-1] * total
        depth[0] = 0
        queue = [0]
        qi = 0

        while qi < len(queue):
            u = queue[qi]
            qi += 1
            d = depth[u]
            pu = parent[u]

            for g in gens:
                v = (u + g) % total
                if depth[v] < 0:
                    depth[v] = d + 1
                    parent[v] = u
                    queue.append(v)
                else:
                    if v == pu or parent[v] == u:
                        continue
                    if d + 1 + depth[v] < 10:
                        return False

        return True

    for x in range(1, m):
        if time.time() > deadline:
            return None
        for y in range(n):
            if time.time() > deadline:
                return None

            if (x == 1 and y == 0) or (x == m - 1 and y == 0):
                continue
            if (2 * x) % m == 0 and (2 * y) % n == 0:
                continue

            if check_girth_fast(x, y):
                edges = []
                for u in range(m):
                    for v in range(n):
                        idx = u * n + v
                        nidx = ((u + 1) % m) * n + v
                        if idx < nidx:
                            edges.append((idx, nidx))
                        nidx = ((u + x) % m) * n + ((v + y) % n)
                        if idx < nidx:
                            edges.append((idx, nidx))
                return total, edges

    return None

def _luw_construct(k, g):
    """
    LUW construction for k-regular graphs with large girth.
    """
    q = k
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)

    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add = lambda x, y: x ^ y
        sub = lambda x, y: x ^ y
        mul = lambda x, y: table[x][y]
    else:
        add = lambda x, y: (x + y) % q
        sub = lambda x, y: (x - y) % q
        mul = lambda x, y: x * y % q

    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r, c0 = (c - 5) % 4, c - (c - 5) % 4
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]

    def neighbours(vertex):
        side, x = vertex
        for first in range(q):
            y = [None, first] + [0] * (d - 1)
            for c in range(2, d + 1):
                a, b = rule[c]
                y[c] = add(x[c], mul(y[a], x[b])) if side == 0 else sub(x[c], mul(x[a], y[b]))
            yield 1 - side, tuple(y)

    start = (0, (None,) + (0,) * d)
    index, order, edges = {start: 0}, [start], []
    for vertex in order:
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))

    return len(order), edges
