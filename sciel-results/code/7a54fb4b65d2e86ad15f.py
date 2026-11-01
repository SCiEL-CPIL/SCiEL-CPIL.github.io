import time
import random
import sys

def construct(k, g):
    # Fallback: Lazebnik-Ustimenko-Woldar graph (large but valid)
    def luw_construct(k, g, d_override=None):
        q = k
        if d_override is not None:
            d = d_override
        else:
            d = max(3, g - 5 if (g - 5) % 2 else g - 4)
        
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

    # Moore bound calculation (using the code's definition)
    D = (g - 2) // 2
    moore_bound = 1 + k * sum((k - 1) ** i for i in range(D))

    deadline = time.time() + 110
    best_n = float('inf')
    best_edges = None

    def has_girth_at_least(n, edges, g):
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        edge_set = set()
        for u, v in edges:
            if u == v:
                return False
            if (v, u) in edge_set:
                return False
            edge_set.add((u, v))
        for u in range(n):
            if len(adj[u]) != k:
                return False
        for start in range(n):
            dist = [-1] * n
            parent = [-1] * n
            dist[start] = 0
            queue = [start]
            head = 0
            while head < len(queue):
                u = queue[head]
                head += 1
                if dist[u] >= g // 2:
                    continue
                for v in adj[u]:
                    if v == parent[u]:
                        continue
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        parent[v] = u
                        queue.append(v)
                    else:
                        if dist[u] + dist[v] + 1 < g:
                            return False
        return True

    def random_regular_graph(n, k):
        stubs = []
        for i in range(n):
            for _ in range(k):
                stubs.append(i)
        random.shuffle(stubs)
        edges = []
        valid = True
        seen = set()
        for i in range(0, len(stubs), 2):
            u = stubs[i]
            v = stubs[i + 1]
            if u == v:
                valid = False
                break
            if (u, v) in seen or (v, u) in seen:
                valid = False
                break
            seen.add((u, v))
            edges.append((u, v))
        if valid:
            return edges
        return None

    # Try LUW with smaller d first
    # Standard d for g=10 is 5 (N=2048). Try d=4 (N=512).
    for d_try in [4, 5]:
        if time.time() > deadline:
            break
        n_luw, edges_luw = luw_construct(k, g, d_try)
        if has_girth_at_least(n_luw, edges_luw, g):
            if n_luw < best_n:
                best_n = n_luw
                best_edges = edges_luw
            # If we found a valid graph with d=4, it's likely the best LUW we can get.
            # But we might find a smaller one via search? Unlikely for 320.
            # If d=4 works, return it immediately? 
            # The prompt says 320 is smallest known. 512 > 320.
            # So if we find 320, we prefer it.
            # But finding 320 is hard.
            # So we hold best_n.
    
    # Search for n=320
    if time.time() < deadline - 20:
        attempts = 500
        for _ in range(attempts):
            if time.time() > deadline:
                break
            edges = random_regular_graph(320, k)
            if edges is None:
                continue
            if has_girth_at_least(320, edges, g):
                return 320, edges

    if best_n != float('inf'):
        return best_n, best_edges
    
    # Fallback to standard LUW
    return luw_construct(k, g)
