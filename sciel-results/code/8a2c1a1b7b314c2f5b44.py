"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import random
import time
from collections import deque

# EVOLVE-BLOCK-START
def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110  # Leave 10s margin

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

    if k != 4 or g != 10:
        return _luw_construct(k, g)

    # Heuristic search for 4-regular graph of girth 10
    def has_girth_at_least(girth, adj, n):
        max_depth = (girth - 1) // 2
        for start in range(n):
            dist = [-1] * n
            parent = [-1] * n
            dist[start] = 0
            queue = deque([start])
            while queue:
                u = queue.popleft()
                if dist[u] >= max_depth:
                    continue
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        parent[v] = u
                        queue.append(v)
                    elif v != parent[u] and dist[v] <= dist[u]:
                        cycle_len = dist[u] + dist[v] + 1
                        if cycle_len < girth:
                            return False
        return True

    def random_4_reg(n):
        half_edges = [i for i in range(n) for _ in range(4)]
        random.shuffle(half_edges)
        adj = [[] for _ in range(n)]
        edges = []
        deg = [0] * n
        for i in range(n):
            for _ in range(4):
                candidates = [j for j in range(n) if deg[j] < 4 and j != i]
                if not candidates:
                    return None
                j = random.choice(candidates)
                if (i, j) in edges or (j, i) in edges:
                    return None
                edges.append((i, j))
                deg[i] += 1
                deg[j] += 1
                adj[i].append(j)
                adj[j].append(i)
        return adj, edges

    def swap_edges(adj, edges, n):
        if len(edges) < 2:
            return None
        e1_idx, e2_idx = random.sample(range(len(edges)), 2)
        u, v = edges[e1_idx]
        x, y = edges[e2_idx]
        
        if u == x or u == y or v == x or v == y:
            return None
        if (u, x) in edges or (x, u) in edges:
            return None
        if (v, y) in edges or (y, v) in edges:
            return None
            
        edges[e1_idx] = (u, x)
        edges[e2_idx] = (v, y)
        
        adj[u].remove(v)
        adj[v].remove(u)
        adj[x].remove(y)
        adj[y].remove(x)
        
        adj[u].append(x)
        adj[x].append(u)
        adj[v].append(y)
        adj[y].append(v)
        
        return edges

    best_n = None
    best_edges = None
    
    for n in range(300, 1000, 10):
        if time.time() > deadline:
            break
        attempts = 0
        found = False
        while attempts < 50 and time.time() < deadline:
            attempts += 1
            res = random_4_reg(n)
            if not res:
                continue
            adj, edges = res
            if has_girth_at_least(10, adj, n):
                best_n = n
                best_edges = edges
                found = True
                break
            
            current_adj, current_edges = adj, edges
            for _ in range(1000):
                if time.time() > deadline:
                    break
                new_edges = swap_edges(current_adj, current_edges, n)
                if new_edges:
                    if has_girth_at_least(10, current_adj, n):
                        best_n = n
                        best_edges = current_edges
                        found = True
                        break
            if found:
                break
    
    if best_n is not None:
        return best_n, best_edges
    else:
        return _luw_construct(4, 10)
# EVOLVE-BLOCK-END
