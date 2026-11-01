import time
import random
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 105  # 15s margin before 120s limit

    # Moore bound for lower bound
    if g % 2 == 0:
        r = g // 2
        moore = k + k * (k - 1) * sum((k - 1) ** (i - 1) for i in range(r))
    else:
        r = (g - 1) // 2
        moore = k * sum((k - 1) ** i for i in range(r))

    # For k=5, g=9: try to find a small graph
    # The LUW construction gives 6250 vertices.
    # Try random greedy construction to find smaller.

    if k == 5 and g == 9:
        # Try multiple random restarts with greedy edge addition
        best_n = 6250
        best_edges = None

        for attempt in range(20):
            if time.time() > deadline:
                break
            t0 = time.time()

            # Try different sizes
            for n in [100, 150, 200, 300, 500]:
                if time.time() > deadline:
                    break
                if n < moore:
                    continue

                # Random greedy: try to build k-regular graph with girth >= g
                # Using a random permutation-based construction
                # Start with empty graph, add edges one by one
                # Ensure degree constraint and girth constraint

                # Alternative: use a random regular graph and check girth
                # For small n, we can try random regular graphs

                if n > 200:
                    continue  # Too slow for random regular graph generation

                # Generate random k-regular graph on n vertices
                # Using configuration model
                # Each vertex has k stubs
                stubs = []
                for v in range(n):
                    for _ in range(k):
                        stubs.append(v)

                random.shuffle(stubs)

                # Pair up stubs to form edges
                edges_set = set()
                valid = True
                for i in range(0, n * k, 2):
                    u, v = stubs[i], stubs[i + 1]
                    if u == v:
                        valid = False
                        break
                    e = (min(u, v), max(u, v))
                    if e in edges_set:
                        valid = False
                        break
                    edges_set.add(e)

                if not valid:
                    continue

                # Check girth
                # BFS from each vertex to find shortest cycle
                # This is expensive, so only do it for small n
                girth = _compute_girth(n, list(edges_set))
                if girth >= g:
                    return n, list(edges_set)

                # If not, try to improve by swapping edges
                # But this is complex, so skip for now

        # If no small graph found, fall back to LUW
        return _luw_construct(k, g)

    # For other parameters, use LUW
    return _luw_construct(k, g)


def _compute_girth(n, edges):
    """Compute girth of a graph using BFS from each vertex."""
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)

    min_girth = float('inf')
    for start in range(n):
        # BFS to find shortest cycle through start
        dist = [-1] * n
        parent = [-1] * n
        dist[start] = 0
        queue = deque([start])

        while queue:
            u = queue.popleft()
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    parent[v] = u
                    queue.append(v)
                elif v != parent[u]:
                    # Found a cycle
                    cycle_len = dist[u] + dist[v] + 1
                    if cycle_len < min_girth:
                        min_girth = cycle_len

    return min_girth if min_girth != float('inf') else n + 1


def _luw_construct(k, g):
    """Lazebnik-Ustimenko-Woldar graph construction."""
    q = k

    # Determine d
    d_odd = g - 5
    if d_odd < 3:
        d_odd = 3
    if d_odd % 2 == 0:
        d_odd += 1

    d_even = g - 4
    if d_even < 4:
        d_even = 4
    if d_even % 2 == 1:
        d_even += 1

    d = min(d_odd, d_even)

    # Field operations
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add = lambda x, y: x ^ y
        sub = lambda x, y: x ^ y
        mul = lambda x, y: table[x][y]
    else:
        add = lambda x, y: (x + y) % q
        sub = lambda x, y: (x - y) % q
        mul = lambda x, y: x * y % q

    # Rule for CD(d, q)
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
                if side == 0:
                    y[c] = add(x[c], mul(y[a], x[b]))
                else:
                    y[c] = sub(x[c], mul(x[a], y[b]))
            yield 1 - side, tuple(y)

    start = (0, (None,) + (0,) * d)
    index = {start: 0}
    order = [start]
    edges = []

    # BFS
    i = 0
    while i < len(order):
        vertex = order[i]
        i += 1
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))

    return len(order), edges
