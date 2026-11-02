import time
from collections import deque

def _verify_girth(n, edges, k, g):
    """Return True if graph is k-regular, simple, and has girth >= g."""
    # Build adjacency
    adj = [[] for _ in range(n)]
    eset = set()
    for u, v in edges:
        if u == v:
            return False
        if u >= n or v >= n:
            return False
        key = (min(u, v), max(u, v))
        if key in eset:
            return False
        eset.add(key)
        adj[u].append(v)
        adj[v].append(u)
    # Check degree
    for u in range(n):
        if len(adj[u]) != k:
            return False
    # BFS from each vertex to find shortest cycle
    # For each vertex s, BFS with parent tracking; first time we reach a
    # vertex with a different parent than the BFS tree edge gives a cycle.
    for s in range(n):
        # BFS
        dist = [-1] * n
        parent = [-1] * n
        q = deque([s])
        dist[s] = 0
        found = False
        while q:
            u = q.popleft()
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    parent[v] = u
                    q.append(v)
                elif parent[u] != v and parent[v] != u:
                    # Found a cycle: u-v edge where neither is parent of other
                    cyc = dist[u] + dist[v] + 1
                    if cyc < g:
                        return False
        # Also check: when we relax an edge to an already-visited vertex
        # that is not the parent, cycle length = dist[u]+dist[v]+1
    # Second pass: more careful cycle detection
    for s in range(n):
        dist = [-1] * n
        parent = [-1] * n
        q = deque([s])
        dist[s] = 0
        while q:
            u = q.popleft()
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    parent[v] = u
                    q.append(v)
                elif parent[u] != v:
                    # v already visited, and u is not v's parent
                    # cycle: s -> ... -> v -> ... -> u -> v (back edge)
                    # length = dist[u] + dist[v] + 1
                    cyc = dist[u] + dist[v] + 1
                    if cyc < g:
                        return False
    return True


def _luw_construct(k, g):
    """LUW construction. Returns (n, edges, d_used)."""
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
    return len(order), edges, d


def _luw_with_d(k, g, d):
    """LUW construction with a specific d value."""
    q = k

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


def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110  # safety margin

    # Known small cages
    if (k, g) == (3, 5):
        edges = []
        for i in range(5):
            edges.append((i, (i + 1) % 5))
        inner = [5, 7, 9, 6, 8]
        for i in range(5):
            edges.append((inner[i], inner[(i + 2) % 5]))
        for i in range(5):
            edges.append((i, 5 + i))
        return 10, edges

    elif (k, g) == (3, 6):
        edges = []
        lines = [(0, 1, 2), (0, 3, 4), (0, 5, 6), (1, 3, 5), (1, 4, 6), (2, 3, 6), (2, 4, 5)]
        for i, line in enumerate(lines):
            line_id = 7 + i
            for p in line:
                edges.append((p, line_id))
        return 14, edges

    elif (k, g) in [(3, 7), (3, 8)]:
        adj = [
            [1, 10, 16], [0, 11, 17], [12, 13, 14], [2, 15, 16], [18, 19, 20],
            [4, 21, 22], [5, 23, 24], [6, 25, 26], [27, 28, 29], [8, 3, 4],
            [0, 1, 10], [1, 0, 11], [2, 3, 12], [12, 2, 13], [13, 3, 14],
            [3, 4, 15], [0, 3, 16], [1, 4, 17], [4, 5, 18], [18, 4, 19],
            [19, 5, 20], [20, 6, 21], [5, 6, 22], [22, 7, 23], [23, 7, 24],
            [24, 7, 25], [25, 7, 26], [26, 7, 27], [27, 7, 28], [28, 7, 29],
            [29, 7, 8]
        ]
        edges = []
        for u in range(30):
            for v in adj[u]:
                if u < v:
                    edges.append((u, v))
        return 30, edges

    # For general (k, g): try LUW with different d values, verify girth
    best_n = None
    best_edges = None

    # Try d values from small to large; smaller d = fewer vertices
    # d must be at least 3; try a range of values
    max_d = 8  # cap to avoid huge graphs
    for d in range(3, max_d + 1):
        if time.time() > deadline:
            break
        try:
            n, edges = _luw_with_d(k, g, d)
            # Quick sanity: n should be 2 * k^d
            if n != 2 * (k ** d):
                continue
            # Verify girth (expensive for large n, so only check if n is manageable)
            if n <= 2000:
                if _verify_girth(n, edges, k, g):
                    if best_n is None or n < best_n:
                        best_n = n
                        best_edges = edges
                    # Since d increases n, once we find a valid one, larger d won't help
                    break
            else:
                # Too large to verify quickly; skip
                break
        except Exception:
            continue

    # If no verified construction found, fall back to the original formula
    if best_edges is None:
        d = max(3, g - 5 if (g - 5) % 2 else g - 4)
        if time.time() > deadline:
            d = max(3, d)
        try:
            n, edges, _ = _luw_construct(k, g)
            return n, edges
        except Exception:
            # Last resort: return a trivial valid graph if possible
            # For k=2, a cycle of length g works
            if k == 2:
                n = g
                edges = [(i, (i + 1) % n) for i in range(n)]
                return n, edges
            # Fallback: return LUW with a large d to ensure girth
            d = g
            n, edges = _luw_with_d(k, g, d)
            return n, edges

    return best_n, best_edges
