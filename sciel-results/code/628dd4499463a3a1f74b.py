import time
import math
import random
from collections import defaultdict

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110

    def check_girth_valid(n, adj, min_girth):
        # BFS from each vertex to check for short cycles
        # Optimization: Only need to check up to depth min_girth // 2
        for s in range(n):
            dist = [-1] * n
            parent = [-1] * n
            dist[s] = 0
            queue = [s]
            head = 0
            while head < len(queue):
                u = queue[head]
                head += 1
                if dist[u] >= min_girth // 2:
                    continue
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        parent[v] = u
                        queue.append(v)
                    elif parent[u] != v and parent[v] != u:
                        cyc_len = dist[u] + dist[v] + 1
                        if cyc_len < min_girth:
                            return False
        return True

    def build_adj(n, edges):
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        # Ensure simple
        for i in range(n):
            adj[i] = list(set(adj[i]))
        return adj

    def luw_construct(k, g):
        # LUW(q, d) is (q-1)-regular in some definitions, but seed uses q=k for degree k.
        # We try d values such that girth is at least g.
        # Girth of LUW(q, d) is at least 2d-1.
        # So we need 2d-1 >= g => d >= (g+1)//2.
        best = None
        min_d = (g + 1) // 2
        for d in range(min_d, g + 5):
            if time.time() > deadline:
                break
            try:
                n, edges = _luw_impl(k, d)
                if n == 0: continue
                adj = build_adj(n, edges)
                if any(len(adj[i]) != k for i in range(n)):
                    continue
                if check_girth_valid(n, adj, g):
                    if best is None or n < best[0]:
                        best = (n, edges)
            except Exception:
                pass
        return best

    def _luw_impl(q, d):
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

    # Generalized Quadrangle Q(s, t)
    # Girth 8.
    # Vertices: Points and Lines of Q(s,t).
    # Points: (x, y, z) in GF(s)^3? No, Q(s,t) is defined on GF(s) x GF(s) x GF(s)?
    # Standard Q(s,t) has points (x, y, z) and lines.
    # For simplicity, we only use it if g <= 8.
    def gq_construct(k, g):
        if g > 8:
            return None
        # k = s + 1 = t + 1 => s = t = k - 1
        s = k - 1
        if s < 2:
            return None
        # Number of points = (s+1)(st+1)
        n_points = (s + 1) * (s * s + 1) # st = s*s since s=t
        n_lines = n_points
        n = 2 * n_points
        edges = []
        # This construction is complex to implement correctly from scratch.
        # Skipping for now to avoid bugs. LUW is safer.
        return None

    candidates = []
    
    # Try LUW
    res = luw_construct(k, g)
    if res:
        candidates.append(res)

    # Specific small cases
    if k == 3 and g == 5:
        # Petersen
        n = 10
        edges = [(i, (i+1)%5) for i in range(5)]
        edges += [(5+i, 5+(i+2)%5) for i in range(5)]
        edges += [(i, 5+i) for i in range(5)]
        candidates.append((n, edges))
    if k == 3 and g == 6:
        # Heawood
        n = 14
        edges = []
        lines = [(0,1,2), (0,3,4), (0,5,6), (1,3,5), (1,4,6), (2,3,6), (2,4,5)]
        for i, line in enumerate(lines):
            for p in line:
                edges.append((p, 7+i))
        candidates.append((n, edges))

    best_n, best_edges = None, None
    for n, edges in candidates:
        adj = build_adj(n, edges)
        if any(len(adj[i]) != k for i in range(n)):
            continue
        if check_girth_valid(n, adj, g):
            if best_n is None or n < best_n:
                best_n = n
                best_edges = edges

    if best_edges is None:
        # Fallback to LUW if validation failed
        return luw_construct(k, g)

    return best_n, best_edges
