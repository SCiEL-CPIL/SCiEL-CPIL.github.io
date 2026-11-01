import sys
import time
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 130.0

    if k == 5 and g == 9:
        # Strategy: LUW(q=5, d=4) has girth 9 and n=1250.
        # This is smaller than LUW(5,5) (n=3125) and satisfies the girth constraint.
        try:
            n, edges = construct_luw(5, 4)
            if n is not None and n > 0:
                if verify_girth(n, edges, k, g):
                    return n, edges
        except Exception:
            pass

        # Fallback: LUW(q=5, d=5) has girth 11 and n=3125.
        try:
            n, edges = construct_luw(5, 5)
            if n is not None and n > 0:
                if verify_girth(n, edges, k, g):
                    return n, edges
        except Exception:
            pass

    # Generic fallback
    return construct_default(k, g)

def construct_luw(q, d):
    """Build the LUW CD(d,q) graph and return (n, edges)."""
    if q == 4:
        table = [[0,0,0,0],[0,1,2,3],[0,2,3,1],[0,3,1,2]]
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
            y = [0, first] + [0] * (d - 1)
            for c in range(2, d + 1):
                a, b = rule[c]
                if side == 0:
                    y[c] = add(x[c], mul(y[a], x[b]))
                else:
                    y[c] = sub(x[c], mul(x[a], y[b]))
            yield 1 - side, tuple(y)

    start = (0, (0,) * (d + 1))
    index = {start: 0}
    order = [start]
    edges = []
    
    # Pre-allocate adj list for verification if needed, but we just need edges for now
    # We will build adj in verify_girth
    
    for vertex in order:
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                u_idx = index[vertex]
                v_idx = index[other]
                edges.append((u_idx, v_idx))

    return len(order), edges

def construct_default(k, g):
    q = k
    # Heuristic for d
    if g % 2 == 1:
        d = (g - 1) // 2
    else:
        d = g // 2 - 1
    if d < 3:
        d = 3
        
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
            y = [0, first] + [0] * (d - 1)
            for c in range(2, d + 1):
                a, b = rule[c]
                if side == 0:
                    y[c] = add(x[c], mul(y[a], x[b]))
                else:
                    y[c] = sub(x[c], mul(x[a], y[b]))
            yield 1 - side, tuple(y)

    start = (0, (0,) * (d + 1))
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

def verify_girth(n, edges, k, g):
    """Verify degree and girth using BFS from each edge endpoint."""
    adj = [[] for _ in range(n)]
    edge_set = set()
    for u, v in edges:
        if u == v:
            return False
        adj[u].append(v)
        adj[v].append(u)
        key = (min(u, v), max(u, v))
        if key in edge_set:
            return False
        edge_set.add(key)

    for i in range(n):
        if len(adj[i]) != k:
            return False

    # Girth check: for each edge (u,v), shortest path from u to v avoiding (u,v)
    # If that path length + 1 < g, girth is violated.
    for u, v in edges:
        dist = [-1] * n
        dist[u] = 0
        queue = deque([u])
        found = False
        while queue:
            cur = queue.popleft()
            if cur == v:
                if dist[v] + 1 < g:
                    return False
                found = True
                break
            for nb in adj[cur]:
                if (cur == u and nb == v) or (cur == v and nb == u):
                    continue
                if dist[nb] == -1:
                    dist[nb] = dist[cur] + 1
                    queue.append(nb)
        if not found:
            return False

    return True
