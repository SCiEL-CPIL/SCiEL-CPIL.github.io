import time
import sys
from collections import deque

def construct(k, g):
    t0 = time.time()
    deadline = 115.0

    def certify_girth(n, edges, g):
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
            
        N2 = n * n
        INF = g
        dist = [INF] * N2
        q = deque()
        
        for a in range(n):
            for b in adj[a]:
                dist[a * n + b] = 1
                q.append(a * n + b)
                
        while q:
            s = q.popleft()
            d = dist[s]
            if d >= g - 1:
                continue
            p = s // n
            c = s % n
            rev = c * n + p
            dr = dist[rev]
            if dr != INF:
                if d + dr + 1 < g:
                    return False
            for nb in adj[c]:
                if nb == p:
                    continue
                ns = c * n + nb
                nd = d + 1
                if nd < dist[ns]:
                    dist[ns] = nd
                    q.append(ns)
        return True

    def luw_construct(k, g, d):
        q = k
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

    best_n, best_edges = None, None

    if k == 3 and g == 16:
        d_candidates = [8, 9, 10, 11, 12]
        for d in d_candidates:
            if time.time() - t0 > deadline - 5:
                break
            try:
                n, edges = luw_construct(k, g, d)
            except Exception:
                continue
            ok = True
            deg = [0] * n
            eset = set()
            for u, v in edges:
                if u == v:
                    ok = False
                    break
                key = (min(u, v), max(u, v))
                if key in eset:
                    ok = False
                    break
                eset.add(key)
                deg[u] += 1
                deg[v] += 1
            if not ok:
                continue
            if any(x != k for x in deg):
                continue
            if time.time() - t0 > deadline - 5:
                break
            if certify_girth(n, edges, g):
                if best_n is None or n < best_n:
                    best_n, best_edges = n, edges
                break

    if best_n is None:
        d = max(3, g - 5 if (g - 5) % 2 else g - 4)
        best_n, best_edges = luw_construct(k, g, d)

    return best_n, best_edges
