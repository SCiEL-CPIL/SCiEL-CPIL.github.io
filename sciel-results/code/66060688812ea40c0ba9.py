import time
import random
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 115.0

    def time_left():
        return deadline - time.time()

    # Moore bound
    if g % 2 == 0:
        moore = k + k * (k - 1) ** (g // 2 - 1)
    else:
        if k == 2:
            moore = g
        else:
            h = (g - 1) // 2
            moore = 1 + k * ((k - 1) ** h - 1) // (k - 2) if k > 2 else g

    best_n = None
    best_edges = None

    def verify_girth_fast(n, edges, min_girth):
        if n == 0:
            return True
        adj = [[] for _ in range(n)]
        edge_set = set()
        for u, v in edges:
            if u == v:
                return False
            key = (min(u, v), max(u, v))
            if key in edge_set:
                return False
            edge_set.add(key)
            adj[u].append(v)
            adj[v].append(u)
        for i in range(n):
            if len(adj[i]) != k:
                return False
        for start in range(n):
            dist = [-1] * n
            parent = [-1] * n
            q = deque([start])
            dist[start] = 0
            while q:
                u = q.popleft()
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        parent[v] = u
                        q.append(v)
                    elif v != parent[u]:
                        cycle_len = dist[u] + dist[v] + 1
                        if cycle_len < min_girth:
                            return False
        return True

    def try_explicit_construction():
        # LUW Construction for k=5, g=9
        # q = k = 5, d = g - 5 = 4
        q = k
        d = g - 5
        if d < 3:
            d = 3

        if q == 4:
            table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
            def add(x, y): return x ^ y
            def sub(x, y): return x ^ y
            def mul(x, y): return table[x][y]
        else:
            def add(x, y): return (x + y) % q
            def sub(x, y): return (x - y) % q
            def mul(x, y): return x * y % q

        # Rule: for c in 2..d, rule[c] = (a, b)
        # Standard LUW rules for k=5, d=4
        rule = {}
        rule[2] = (1, 1)
        rule[3] = (2, 1)
        rule[4] = (1, 2)

        def neighbours(vertex):
            side, x = vertex
            for first in range(q):
                y = [None, first] + [0] * (d - 1)
                for c in range(2, d + 1):
                    a, b = rule[c]
                    if side == 0:
                        y[c] = add(x[c], mul(y[a], x[b]))
                    else:
                        y[c] = sub(x[c], mul(x[a], y[b]))
                yield 1 - side, tuple(y)

        start = (0, (None,) + (0,) * d)
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

    # Try LUW first
    if time_left() > 10:
        try:
            n_exp, edges_exp = try_explicit_construction()
            if verify_girth_fast(n_exp, edges_exp, g):
                best_n = n_exp
                best_edges = edges_exp
        except Exception:
            pass

    # Random Search with Repair
    if best_n is None or best_n > 1000:
        if time_left() > 10:
            # Search for smaller n
            search_start = max(moore, 100)
            search_end = best_n if best_n else 2000
            step = 50
            
            for n_try in range(search_start, search_end + 1, step):
                if time_left() < 5:
                    break
                for attempt in range(100):
                    if time_left() < 5:
                        break
                    # Generate random regular graph
                    # Use configuration model
                    stubs = [i for i in range(n_try) for _ in range(k)]
                    random.shuffle(stubs)
                    adj = [[] for _ in range(n_try)]
                    edge_set = set()
                    valid = True
                    for i in range(n_try * k // 2):
                        u, v = stubs[2 * i], stubs[2 * i + 1]
                        if u == v:
                            valid = False
                            break
                        key = (min(u, v), max(u, v))
                        if key in edge_set:
                            valid = False
                            break
                        edge_set.add(key)
                        adj[u].append(v)
                        adj[v].append(u)
                    
                    if not valid:
                        continue
                        
                    edges = list(edge_set)
                    if verify_girth_fast(n_try, edges, g):
                        if best_n is None or n_try < best_n:
                            best_n = n_try
                            best_edges = edges
                        break

    if best_n is None:
        # Fallback to LUW even if verify failed? No.
        # Return trivial if possible?
        if k == 2:
            n = g
            edges = [(i, (i+1)%n) for i in range(n)]
            return n, edges
        return 0, []

    return best_n, best_edges
