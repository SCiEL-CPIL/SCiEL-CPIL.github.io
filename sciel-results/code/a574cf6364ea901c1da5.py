import time
import math
import random
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 145.0  # 5s margin for safety

    best_n = float('inf')
    best_edges = None

    def verify_graph(n, edges, k, g, time_limit=10.0):
        if time.time() > start_time + time_limit:
            return False

        if n == 0:
            return False
        
        adj = [[] for _ in range(n)]
        seen_edges = set()
        
        for u, v in edges:
            if u == v:
                return False
            if u < 0 or u >= n or v < 0 or v >= n:
                return False
            edge_key = (min(u, v), max(u, v))
            if edge_key in seen_edges:
                return False
            seen_edges.add(edge_key)
            adj[u].append(v)
            adj[v].append(u)
        
        for i in range(n):
            if len(adj[i]) != k:
                return False
        
        max_depth = g // 2
        
        for u in range(n):
            if time.time() > start_time + time_limit:
                return False
                
            dist = [-1] * n
            parent = [-1] * n
            dist[u] = 0
            q = deque([u])
            
            while q:
                curr = q.popleft()
                if dist[curr] >= max_depth:
                    continue
                
                for v in adj[curr]:
                    if dist[v] == -1:
                        dist[v] = dist[curr] + 1
                        parent[v] = curr
                        q.append(v)
                    else:
                        if v == parent[curr]:
                            continue
                        cycle_len = dist[curr] + 1 + dist[v]
                        if cycle_len < g:
                            return False
                            
        return True

    def build_luw_graph(q_val, d_val):
        if q_val == 4:
            table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
            add, sub, mul = (lambda x, y: x ^ y), (lambda x, y: x ^ y), (lambda x, y: table[x][y])
        else:
            add, sub, mul = (lambda x, y: (x + y) % q_val), (lambda x, y: (x - y) % q_val), (lambda x, y: x * y % q_val)

        rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
        
        def get_rule(d_v):
            r = dict(rule)
            for c in range(5, d_v + 1):
                rc, c0 = (c - 5) % 4, c - (c - 5) % 4
                r[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][rc]
            return r

        r = get_rule(d_val)
        
        def neighbours(vertex):
            side, x = vertex
            for first in range(q_val):
                y = [None, first] + [0] * (d_val - 1)
                for c in range(2, d_val + 1):
                    a, b = r[c]
                    y[c] = add(x[c], mul(y[a], x[b])) if side == 0 else sub(x[c], mul(x[a], y[b]))
                yield 1 - side, tuple(y)

        start = (0, (None,) + (0,) * d_val)
        index, order, edges = {start: 0}, [start], []
        for vertex in order:
            for other in neighbours(vertex):
                if other not in index:
                    index[other] = len(order)
                    order.append(other)
                if vertex[0] == 0:
                    edges.append((index[vertex], index[other]))
        return len(order), edges

    # Corrected min_d calculation: Girth of LUW is 2d+2.
    # We need 2d+2 >= g => d >= (g-2)/2
    min_d = max(3, (g - 2) // 2 + (1 if (g - 2) % 2 != 0 else 0))
    
    # Try LUW family
    q = k
    for d_val in range(min_d, min_d + 15):
        if time.time() > deadline - 30:
            break
        try:
            n, edges = build_luw_graph(q, d_val)
            if n > 50000:
                continue
            if n < best_n:
                t_limit = 5.0 if n > 10000 else 20.0
                if verify_graph(n, edges, k, g, time_limit=t_limit):
                    best_n = n
                    best_edges = edges
                    break
        except Exception:
            pass

    if best_edges is None:
        return 0, []

    return best_n, best_edges
