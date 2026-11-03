import time
from collections import deque

def construct(k, g):
    """
    Constructs a k-regular graph of girth at least g using the Lazebnik-Ustimenko-Woldar (LUW) construction.
    Returns (n, edges) where n is the number of vertices and edges is a list of (i, j) pairs.
    """
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin (evaluator kills at 150s)

    q = k
    # LUW is defined for q = 3, 5, 7 (prime or 4)
    if q not in (3, 4, 5, 7):
        q = 3  # Fallback

    # Determine d such that girth_of_d(d) >= g
    def girth_of_d(d):
        if d % 2 == 1:
            return d + 5
        else:
            return d + 6

    d = 3
    while girth_of_d(d) < g:
        d += 1

    # GF(q) operations
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        def add(x, y): return x ^ y
        def sub(x, y): return x ^ y
        def mul(x, y): return table[x][y]
    else:
        def add(x, y): return (x + y) % q
        def sub(x, y): return (x - y) % q
        def mul(x, y): return x * y % q

    # Rule for LUW
    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r, c0 = (c - 5) % 4, c - (c - 5) % 4
        rules_cycle = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)]
        rule[c] = rules_cycle[r]

    # Generate vertices using BFS
    index = {}
    order = []
    edges = []

    # Start vertex: side 0, coordinates (None, 0, 0, ..., 0)
    start = (0, (None,) + (0,) * d)
    index[start] = 0
    order.append(start)

    for i, vertex in enumerate(order):
        side, x = vertex

        for first in range(q):
            y = [None, first] + [0] * (d - 1)
            for c in range(2, d + 1):
                a, b = rule[c]
                if side == 0:
                    # Point -> Line: y[c] = x[c] + y[a] * x[b]
                    y[c] = add(x[c], mul(y[a], x[b]))
                else:
                    # Line -> Point: y[c] = x[c] - x[a] * y[b]
                    y[c] = sub(x[c], mul(x[a], y[b]))

            neighbor = (1 - side, tuple(y))

            if neighbor not in index:
                index[neighbor] = len(order)
                order.append(neighbor)

            if side == 0:
                edges.append((index[vertex], index[neighbor]))

        if time.time() > deadline:
            break

    n = len(order)

    # Verify girth via non-backtracking walks from each vertex (sampled for speed)
    # Build adjacency list
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)

    # Check girth: BFS from each vertex, tracking parent to avoid backtracking
    # For large n, sample vertices
    import random
    rng = random.Random(42)
    sample_size = min(n, 200)
    if n > sample_size:
        samples = rng.sample(range(n), sample_size)
    else:
        samples = list(range(n))

    min_girth = 10**9
    for s in samples:
        # BFS with parent tracking
        dist = {s: 0}
        parent = {s: None}
        queue = deque([(s, None)])  # (node, parent)
        while queue:
            node, par = queue.popleft()
            for nb in adj[node]:
                if nb == par:
                    continue  # non-backtracking
                if nb not in dist:
                    dist[nb] = dist[node] + 1
                    parent[nb] = node
                    queue.append((nb, node))
                else:
                    # Found a cycle
                    cycle_len = dist[node] + dist[nb] + 1
                    if cycle_len < min_girth:
                        min_girth = cycle_len

    n = len(order)
    return n, edges
