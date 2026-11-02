import time
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110.0

    # Petersen graph: 3-regular, girth 5, n=10
    if k == 3 and g == 5:
        return _petersen_graph()

    # For the main task: 5-regular, girth 9
    if k == 5 and g == 9:
        n, edges = _luw_construct(5, 4)
        if n > 0 and _verify_girth_fast(n, edges, g, k):
            return n, edges

    # General LUW construction
    for d in range(2, 12):
        if time.time() > deadline - 5:
            break
        n, edges = _luw_construct(k, d)
        if n > 0 and _verify_girth_fast(n, edges, g, k):
            return n, edges

    # Fallback
    d = 12
    n, edges = _luw_construct(k, d)
    return n, edges

def _petersen_graph():
    edges = []
    for i in range(5):
        edges.append((i, (i + 1) % 5))
        edges.append((5 + i, 5 + (i + 2) % 5))
        edges.append((i, 5 + i))
    return 10, edges

def _luw_construct(k, d):
    """
    Lazebnik-Ustimenko-Woldar CD(d, q) with q = k.
    Returns a k-regular bipartite graph on 2*k^d vertices.
    """
    q = k
    if q < 2 or d < 1:
        return 0, []

    mul_table = [[(i * j) % q for j in range(q)] for i in range(q)]

    rule = {}
    if d >= 2:
        rule[2] = (1, 1)
    if d >= 3:
        rule[3] = (2, 1)
    if d >= 4:
        rule[4] = (1, 2)
    for c in range(5, d + 1):
        r = (c - 5) % 4
        c0 = c - r
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
        for first in range(q):
            y = [0] * d
            y[0] = first
            for c in range(1, d):
                cc = c + 1
                a, b = rule[cc]
                a_idx = a - 1
                b_idx = b - 1
                if side == 0:
                    y[c] = (x[c] + mul_table[y[a_idx]][x[b_idx]]) % q
                else:
                    y[c] = (x[c] - mul_table[x[a_idx]][y[b_idx]]) % q
            yield 1 - side, tuple(y)

    start = (0, tuple([0] * d))
    index = {start: 0}
    order = [start]
    edges = []

    for i, vertex in enumerate(order):
        side, x = vertex
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if side == 0:
                edges.append((index[vertex], index[other]))

    return len(order), edges

def _verify_girth_fast(n, edges, g, k):
    """
    Verify that the graph is k-regular and has no cycle shorter than g.
    """
    if n == 0:
        return False

    adj = [[] for _ in range(n)]
    for u, v in edges:
        if u == v:
            return False
        if u < 0 or u >= n or v < 0 or v >= n:
            return False
        adj[u].append(v)
        adj[v].append(u)

    # Check degrees
    for i in range(n):
        if len(adj[i]) != k:
            return False

    # BFS-based girth check
    for start in range(n):
        dist = [-1] * n
        parent = [-1] * n
        dist[start] = 0
        queue = deque([start])

        while queue:
            u = queue.popleft()
            du = dist[u]
            if du >= g // 2:
                continue
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = du + 1
                    parent[v] = u
                    queue.append(v)
                elif v != parent[u]:
                    cycle_len = du + dist[v] + 1
                    if cycle_len < g:
                        return False
    return True

def _verify_girth(n, edges, g):
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)

    for start in range(n):
        dist = [-1] * n
        parent = [-1] * n
        dist[start] = 0
        queue = deque([start])

        while queue:
            u = queue.popleft()
            du = dist[u]
            if du >= g // 2:
                continue
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = du + 1
                    parent[v] = u
                    queue.append(v)
                elif v != parent[u]:
                    cycle_len = du + dist[v] + 1
                    if cycle_len < g:
                        return False
    return True
