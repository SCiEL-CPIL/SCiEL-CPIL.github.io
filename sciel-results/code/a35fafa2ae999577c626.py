import numpy as np
from collections import deque
import time

def construct(k, g):
    start_time = time.time()
    time_limit = 115  # leave 5s buffer
    
    if k == 4 and g == 10:
        # Try random 4-regular graphs with increasing n
        # Start from known lower bound area
        for n in range(320, 321):
            attempts = 0
            while time.time() - start_time < time_limit and attempts < 500:
                edges = generate_random_k_regular(n, k)
                if edges is None:
                    attempts += 1
                    continue
                adj = build_adj(n, edges)
                if check_girth(adj, g):
                    return n, edges
                attempts += 1
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
    """Generate a random k-regular graph using configuration model with proper retry."""
    if n * k % 2 != 0:
        return None
    
    max_attempts = 50
    for _ in range(max_attempts):
        stubs = [i for i in range(n) for _ in range(k)]
        np.random.shuffle(stubs)
        edges = set()
        valid = True
        for i in range(0, len(stubs), 2):
            u, v = stubs[i], stubs[i+1]
            if u == v:
                valid = False
                break
            if (u, v) in edges or (v, u) in edges:
                valid = False
                break
            edges.add((min(u, v), max(u, v)))
        if valid and len(edges) == n * k // 2:
            return list(edges)
    return None

def build_adj(n, edges):
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    return adj

def check_girth(adj, g):
    """Check if graph has girth at least g using BFS from each vertex.
    A cycle of length < g exists iff there's an edge (u,v) where dist[u][v] + 1 < g
    and u != v, considering the shortest path not using that edge."""
    n = len(adj)
    for u in range(n):
        dist = [-1] * n
        dist[u] = 0
        q = deque([u])
        while q:
            v = q.popleft()
            if dist[v] >= g - 1:
                break  # No cycle shorter than g can be found from here
            for w in adj[v]:
                if dist[w] == -1:
                    dist[w] = dist[v] + 1
                    q.append(w)
                else:
                    if v != u and dist[v] + dist[w] + 1 < g:
                        return False
    return True
