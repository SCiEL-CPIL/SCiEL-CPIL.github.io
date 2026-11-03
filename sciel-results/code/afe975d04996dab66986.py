import time
from collections import deque

def _check_girth_nb(adj, n, g):
    for start in range(n):
        for first in adj[start]:
            visited = set()
            q = deque()
            q.append((first, start, 0))
            visited.add((first, start))
            bad = False
            while q:
                curr, prev, depth = q.popleft()
                if depth >= g - 1:
                    continue
                for nxt in adj[curr]:
                    if nxt == prev:
                        continue
                    if nxt == start and depth + 1 < g:
                        bad = True
                        break
                    st = (nxt, curr)
                    if st not in visited:
                        visited.add(st)
                        q.append((nxt, curr, depth + 1))
                if bad:
                    break
            if bad:
                return False
    return True

def _luw(k, g):
    q = k
    d = g - 5
    if d < 3:
        d = 3
    if q == 4:
        tbl = [[0,0,0,0],[0,1,2,3],[0,2,3,1],[0,3,1,2]]
        add = sub = (lambda x, y: x ^ y)
        mul = (lambda x, y: tbl[x][y])
    else:
        add = (lambda x, y: (x + y) % q)
        sub = (lambda x, y: (x - y) % q)
        mul = (lambda x, y: (x * y) % q)
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
            yield (1 - side, tuple(y))

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

def _woldar_gen(q, d):
    # Woldar construction: (q, 2d+2)-cage family.
    if q == 4:
        tbl = [[0,0,0,0],[0,1,2,3],[0,2,3,1],[0,3,1,2]]
        add = sub = (lambda x, y: x ^ y)
        mul = (lambda x, y: tbl[x][y])
    else:
        add = (lambda x, y: (x + y) % q)
        sub = (lambda x, y: (x - y) % q)
        mul = (lambda x, y: (x * y) % q)
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
            yield (1 - side, tuple(y))

    start = (0, (None,) + (0,) * d)
    index, order, edges = {start: 0}, [start], []
    for vertex in order:
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
    n = len(order)
    deg = [0] * n
    for u, v in edges:
        deg[u] += 1
        deg[v] += 1
    assert all(dd == q for dd in deg), deg
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    return n, edges, adj

def _verify_valid(n, edges, k, g):
    """Verify: simple, k-regular, girth >= g."""
    if n < 1:
        return False
    adj = [[] for _ in range(n)]
    edge_set = set()
    for u, v in edges:
        if u == v:
            return False
        if u >= n or v >= n or u < 0 or v < 0:
            return False
        key = (min(u,v), max(u,v))
        if key in edge_set:
            return False
        edge_set.add(key)
        adj[u].append(v)
        adj[v].append(u)
    for i in range(n):
        if len(adj[i]) != k:
            return False
    # Girth check via non-backtracking BFS
    for start in range(n):
        for first in adj[start]:
            visited = set()
            q = deque()
            q.append((first, start, 1))
            visited.add((first, start))
            while q:
                curr, prev, depth = q.popleft()
                if depth >= g:
                    continue
                for nxt in adj[curr]:
                    if nxt == prev:
                        continue
                    if nxt == start and depth + 1 < g:
                        return False
                    st = (nxt, curr)
                    if st not in visited:
                        visited.add(st)
                        q.append((nxt, curr, depth + 1))
    return True

def construct(k, g):
    """
    Constructs a k-regular graph with girth at least g.
    Uses Woldar/Lazebnik-Ustimenko-Woldar algebraic construction.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill

    # For the (4,10) case, use Woldar construction with q=4, d=4
    # which gives a 4-regular graph with girth 2*4+2 = 10.
    if k == 4 and g == 10:
        n, edges, adj = _woldar_gen(4, 4)
        # Verify it's valid
        if _verify_valid(n, edges, 4, 10):
            return n, edges
        # If d=4 doesn't work, try d=5 (girth 12, still >= 10)
        n, edges, adj = _woldar_gen(4, 5)
        if _verify_valid(n, edges, 4, 10):
            return n, edges
        # Fallback
        n, edges = _luw(4, 10)
        return n, edges

    # Fallback for other (k, g)
    if g <= 2:
        n = max(k, 2)
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                edges.append((i, j))
        return n, edges
    if g == 4:
        # k-regular graph with girth >= 4 means no triangles
        # Use complete bipartite K_{k,k} if k >= 2
        n = 2 * k
        edges = []
        for i in range(k):
            for j in range(k):
                edges.append((i, k + j))
        return n, edges

    # General fallback: LUW construction
    n, edges = _luw(k, g)
    return n, edges
