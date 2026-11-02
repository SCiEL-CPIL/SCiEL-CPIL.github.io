import random
import time
from collections import deque

def construct(k, g):
    start_time = time.time()
    time_limit = 115.0
    
    # Fallback LUW construction
    def luw_construct():
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

    # Check if k, g is the specific case
    if k == 5 and g == 9:
        # Try search
        def is_valid_graph(n, edges):
            if len(edges) != k * n // 2:
                return False
            edge_set = set()
            for u, v in edges:
                if u == v:
                    return False
                key = (min(u, v), max(u, v))
                if key in edge_set:
                    return False
                edge_set.add(key)
            
            degrees = [0] * n
            for u, v in edges:
                degrees[u] += 1
                degrees[v] += 1
            if any(d != k for d in degrees):
                return False
                
            adj = [[] for _ in range(n)]
            for u, v in edges:
                adj[u].append(v)
                adj[v].append(u)
                
            min_girth = float('inf')
            for start in range(n):
                dist = [-1] * n
                parent = [-1] * n
                dist[start] = 0
                q = deque([start])
                
                while q:
                    u = q.popleft()
                    for v in adj[u]:
                        if dist[v] == -1:
                            dist[v] = dist[u] + 1
                            parent[v] = u
                            q.append(v)
                        elif parent[u] != v and parent[v] != u:
                            cycle_len = dist[u] + dist[v] + 1
                            if cycle_len < min_girth:
                                min_girth = cycle_len
                                if min_girth < g:
                                    return False
            return True

        def generate_random_graph(n):
            while True:
                half_edges = []
                for i in range(n):
                    for _ in range(k):
                        half_edges.append(i)
                random.shuffle(half_edges)
                
                edges = []
                valid = True
                seen = set()
                for i in range(0, len(half_edges), 2):
                    u = half_edges[i]
                    v = half_edges[i+1]
                    if u == v:
                        valid = False
                        break
                    key = (min(u, v), max(u, v))
                    if key in seen:
                        valid = False
                        break
                    seen.add(key)
                    edges.append((u, v))
                
                if valid:
                    return edges

        def local_search(n, edges, max_iters=500):
            adj = [[] for _ in range(n)]
            for u, v in edges:
                adj[u].append(v)
                adj[v].append(u)
            
            edge_set = set((min(u, v), max(u, v)) for u, v in edges)
            
            def find_short_cycle():
                for start in range(n):
                    dist = [-1] * n
                    parent = [-1] * n
                    dist[start] = 0
                    q = deque([start])
                    
                    while q:
                        u = q.popleft()
                        for v in adj[u]:
                            if dist[v] == -1:
                                dist[v] = dist[u] + 1
                                parent[v] = u
                                q.append(v)
                            elif parent[u] != v and parent[v] != u:
                                return u, v
                return None

            for _ in range(max_iters):
                if time.time() - start_time > time_limit:
                    return False, edges
                    
                conflict = find_short_cycle()
                if conflict is None:
                    return True, edges
                
                u, v = conflict
                
                # Pick random edges to swap
                e1 = random.choice(edges)
                u1, v1 = e1
                
                e2 = random.choice(edges)
                u2, v2 = e2
                
                # Try swapping
                new_edges = [e for e in edges if e != e1 and e != e2]
                
                # Option 1
                if u1 != u2 and v1 != v2:
                    ne1 = (min(u1, u2), max(u1, u2))
                    ne2 = (min(v1, v2), max(v1, v2))
                    if ne1 != ne2 and ne1 not in edge_set and ne2 not in edge_set:
                        edges = new_edges + [ne1, ne2]
                        adj = [[] for _ in range(n)]
                        edge_set = set()
                        for a, b in edges:
                            adj[a].append(b)
                            adj[b].append(a)
                            edge_set.add((min(a, b), max(a, b)))
                        continue
                        
                # Option 2
                if u1 != v2 and v1 != u2:
                    ne1 = (min(u1, v2), max(u1, v2))
                    ne2 = (min(v1, u2), max(v1, u2))
                    if ne1 != ne2 and ne1 not in edge_set and ne2 not in edge_set:
                        edges = new_edges + [ne1, ne2]
                        adj = [[] for _ in range(n)]
                        edge_set = set()
                        for a, b in edges:
                            adj[a].append(b)
                            adj[b].append(a)
                            edge_set.add((min(a, b), max(a, b)))
                        continue
                        
            return False, edges

        # Search loop
        for N in range(428, 450):
            if time.time() - start_time > time_limit:
                break
                
            for _ in range(10):
                if time.time() - start_time > time_limit:
                    break
                    
                edges = generate_random_graph(N)
                if is_valid_graph(N, edges):
                    return N, edges
                    
                success, final_edges = local_search(N, edges, max_iters=200)
                if success:
                    if is_valid_graph(N, final_edges):
                        return N, final_edges
                        
    # Fallback
    return luw_construct()
