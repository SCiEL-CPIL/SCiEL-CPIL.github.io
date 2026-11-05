import time
import sys
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = 120.0
    safety_margin = 5.0
    
    q = k
    
    best_d = None
    best_n = float('inf')
    best_graph = None
    
    for d in range(3, 12):
        if (time.time() - start_time) > deadline - safety_margin:
            break
            
        if d % 2 == 1:
            est_girth = d + 5
        else:
            est_girth = 2 * d + 1
            
        if est_girth < g:
            continue
            
        n_candidate = 2 * (q ** d)
        if n_candidate > 50000:
            continue
            
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
                    
        n = len(order)
        if n != n_candidate:
            continue

        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
            
        is_regular = all(len(adj[i]) == k for i in range(n))
        if not is_regular:
            continue

        r = (g - 1) // 2
        valid_girth = True
        for start_v in range(n):
            if (time.time() - start_time) > deadline - safety_margin:
                valid_girth = False
                break
            visited = [-1] * n
            parent = [-1] * n
            visited[start_v] = 0
            queue = deque([(start_v, 0)])
            
            while queue:
                u, dist = queue.popleft()
                if dist >= r:
                    continue
                for v in adj[u]:
                    if visited[v] == -1:
                        visited[v] = dist + 1
                        parent[v] = u
                        queue.append((v, dist + 1))
                    else:
                        if v != parent[u]:
                            cycle_len = dist[u] + dist[v] + 1
                            if cycle_len < g:
                                valid_girth = False
                                break
                if not valid_girth:
                    break
            
            if not valid_girth:
                break
        
        if valid_girth:
            if n < best_n:
                best_n = n
                best_d = d
                best_graph = (n, edges)
                break

    if best_graph is not None:
        return best_graph
    else:
        return (0, [])
