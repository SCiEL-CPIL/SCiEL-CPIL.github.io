import random
from collections import deque

def construct(k, g):
    def compute_girth_fast(n, adj, target_girth):
        for start in range(n):
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
                    elif parent[u] != v and parent[v] != u:
                        cycle_len = dist[u] + dist[v] + 1
                        if cycle_len < target_girth:
                            return cycle_len
        return target_girth
    
    def random_3_regular(n):
        half_edges = [i for i in range(n) for _ in range(3)]
        random.shuffle(half_edges)
        edges = []
        seen = set()
        for i in range(0, len(half_edges), 2):
            u, v = half_edges[i], half_edges[i+1]
            if u == v:
                return None, None
            key = (min(u,v), max(u,v))
            if key in seen:
                return None, None
            seen.add(key)
            edges.append((u, v))
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        return edges, adj
    
    moore_bound = 2 * sum(2**i for i in range(g//2))
    start_n = max(moore_bound, 4)
    if start_n % 2 != 0:
        start_n += 1
    
    for n in range(start_n, start_n + 50):
        if n % 2 != 0:
            continue
        for attempt in range(10):
            edges, adj = random_3_regular(n)
            if edges is None:
                continue
            girth = compute_girth_fast(n, adj, g)
            if girth >= g:
                return n, edges
    
    # Fallback to LUW
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
