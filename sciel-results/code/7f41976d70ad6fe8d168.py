import random
import time
from collections import deque

def _check_girth(edges, n, g):
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    max_depth = (g - 2) // 2
    
    for s in range(n):
        dist = [-1] * n
        parent = [-1] * n
        dist[s] = 0
        queue = deque([s])
        while queue:
            u = queue.popleft()
            if dist[u] >= max_depth:
                continue
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    parent[v] = u
                    queue.append(v)
                else:
                    if v != parent[u] and u != parent[v]:
                        cycle_len = dist[u] + dist[v] + 1
                        if cycle_len < g:
                            return False
    return True

def _luw_construct(k, g):
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

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 115
    
    if k == 4 and g == 10:
        # Try circulant graphs C(n; 1, a)
        for n in range(242, 350):
            if time.time() > deadline - 5:
                break
            for a in range(2, n // 2 + 1):
                if time.time() > deadline - 5:
                    break
                if a == 1 or a == n - 1:
                    continue
                # Quick check for 4-cycles
                # 4-cycle exists if 2a ≡ ±2, a±1 ≡ ±(a∓1) mod n, etc.
                # Simplified: check if any of 2, 2a, a-1, a+1, n-(a-1), n-(a+1) ≡ 2 mod n
                if (2 * a) % n in (2, n - 2) or (a - 1) % n in (2, n - 2) or (a + 1) % n in (2, n - 2):
                    continue
                edges = []
                for i in range(n):
                    edges.append((i, (i + 1) % n))
                    edges.append((i, (i + a) % n))
                edge_set = set()
                valid = True
                for u, v in edges:
                    if u == v:
                        valid = False
                        break
                    key = (min(u, v), max(u, v))
                    if key in edge_set:
                        valid = False
                        break
                    edge_set.add(key)
                if not valid or len(edge_set) != 2 * n:
                    continue
                unique_edges = list(edge_set)
                if _check_girth(unique_edges, n, g):
                    return n, unique_edges
        
        # Fallback: LUW
        return _luw_construct(k, g)
    else:
        return _luw_construct(k, g)
