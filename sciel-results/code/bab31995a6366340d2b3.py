import time
import random
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 145.0

    # 1. Construct LUW graph as a fallback
    q = k
    d = g - 5
    if d < 3:
        d = 3
    if (d + 5) < g:
        d += 1
        
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

    def get_luw_neighbors(vertex):
        side, x = vertex
        neighbors = []
        for first in range(q):
            y = [None, first] + [0] * (d - 1)
            for c in range(2, d + 1):
                a, b = rule[c]
                if side == 0:
                    y[c] = add(x[c], mul(y[a], x[b]))
                else:
                    y[c] = sub(x[c], mul(x[a], y[b]))
            neighbors.append((1 - side, tuple(y)))
        return neighbors

    luw_index = {}
    luw_order = []
    luw_edges = []
    start_vertex = (0, (None,) + (0,) * d)
    luw_index[start_vertex] = 0
    luw_order.append(start_vertex)
    
    for v in luw_order:
        for u in get_luw_neighbors(v):
            if u not in luw_index:
                luw_index[u] = len(luw_order)
                luw_order.append(u)
            if v[0] == 0:
                luw_edges.append((luw_index[v], luw_index[u]))
                
    best_n = len(luw_order)
    best_edges = luw_edges

    # 2. Try to find a smaller graph using Voltage Graphs on K_{k+1}
    # Base: K_{k+1}
    base_n = k + 1
    
    def is_valid_graph(n, edges, k, g):
        if len(edges) != n * k // 2:
            return False
        deg = [0] * n
        adj = [[] for _ in range(n)]
        for u, v in edges:
            if u == v: return False
            deg[u] += 1
            deg[v] += 1
            adj[u].append(v)
            adj[v].append(u)
        if any(d != k for d in deg):
            return False
            
        # Check girth
        # For each start, BFS
        for start in range(n):
            dist = [-1] * n
            parent = [-1] * n
            dist[start] = 0
            queue = deque([start])
            while queue:
                u = queue.popleft()
                if dist[u] >= g - 1:
                    continue
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        parent[v] = u
                        queue.append(v)
                    elif v != parent[u]:
                        # Cycle found
                        # Length is dist[u] + dist[v] + 1? 
                        # This is an upper bound. If < g, invalid.
                        if dist[u] + dist[v] + 1 < g:
                            return False
        return True

    # Random search
    # We want (k+1)*m < best_n
    max_m = best_n // base_n
    if max_m > 10:
        random.seed(42)
        for _ in range(500):
            if time.time() > deadline - 5:
                break
            m = random.randint(2, max_m)
            # Random voltages
            v = [[random.randint(0, m-1) for _ in range(base_n)] for _ in range(base_n)]
            for i in range(base_n):
                v[i][i] = 0
                
            cover_n = base_n * m
            if cover_n >= best_n:
                continue
                
            cover_edges = []
            for i in range(base_n):
                for a in range(m):
                    for j in range(base_n):
                        if i == j: continue
                        b = (a + v[i][j]) % m
                        u = i * m + a
                        w = j * m + b
                        if u < w:
                            cover_edges.append((u, w))
                            
            if len(cover_edges) != cover_n * k // 2:
                continue
                
            if is_valid_graph(cover_n, cover_edges, k, g):
                best_n = cover_n
                best_edges = cover_edges
                # Don't break, might find smaller
                # But update max_m? No, keep searching for even smaller
                # Actually, if we find one, we can try to find smaller m
                max_m = best_n // base_n

    return best_n, best_edges
