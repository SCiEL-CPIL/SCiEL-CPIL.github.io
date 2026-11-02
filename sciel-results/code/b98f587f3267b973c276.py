import numpy as np
from collections import deque

def construct(k, g):
    # Fallback to LUW if specific case not handled or fails
    if k == 4 and g == 10:
        # Try to find a smaller graph
        for n in range(243, 321):
            # Try a few random graphs
            for _ in range(20):
                edges = generate_random_k_regular(n, k)
                if edges is None:
                    continue
                adj = build_adj(n, edges)
                if check_girth(adj, g):
                    return n, edges
        # If not found, try harder for 320
        for _ in range(200):
            n = 320
            edges = generate_random_k_regular(n, k)
            if edges is None:
                continue
            adj = build_adj(n, edges)
            if check_girth(adj, g):
                return n, edges
        # Fallback to LUW
        return luw_construct(k, g)
    else:
        return luw_construct(k, g)

def luw_construct(k, g):
    # Original LUW code
    q = k
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add, sub, mul = (lambda x, y: x ^ y), (lambda x, y: x ^ y), (lambda x, y: table[x][y])
    else:
        add, sub, mul = (lambda x, y: (x + y) % q), (lambda x, y: (x - y) % q), (lambda x, y: x * y % q)
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

def generate_random_k_regular(n, k):
    # Configuration model
    stubs = [i for i in range(n) for _ in range(k)]
    np.random.shuffle(stubs)
    edges = set()
    for i in range(0, len(stubs), 2):
        u, v = stubs[i], stubs[i+1]
        if u == v:
            continue # Self loop, retry logic needed
        if (u, v) in edges or (v, u) in edges:
            continue # Multi edge
        edges.add((min(u, v), max(u, v)))
    # Check if we have n*k/2 edges
    if len(edges) != n * k // 2:
        return None
    return list(edges)

def build_adj(n, edges):
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    return adj

def check_girth(adj, g):
    n = len(adj)
    for u in range(n):
        dist = [-1] * n
        dist[u] = 0
        q = deque([u])
        while q:
            v = q.popleft()
            for w in adj[v]:
                if dist[w] == -1:
                    dist[w] = dist[v] + 1
                    q.append(w)
                else:
                    # Cycle length
                    if dist[v] + dist[w] + 1 < g:
                        return False
    return True
