import time
import random
import numpy as np

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # Fallback: LUW construction (from seed)
    def luw_fallback():
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

    # Random search for smaller graph
    # Moore bound for k=3, g=16 is 382. Lower bound 512.
    # Try N from 512 upwards.
    
    def check_girth(n, adj, min_girth):
        # Check if girth >= min_girth
        # BFS from each vertex
        for start in range(n):
            visited = [-1] * n
            parent = [-1] * n
            dist = [0] * n
            visited[start] = 0
            queue = [start]
            head = 0
            while head < len(queue):
                u = queue[head]
                head += 1
                for v in adj[u]:
                    if visited[v] == -1:
                        visited[v] = dist[u] + 1
                        parent[v] = u
                        queue.append(v)
                    else:
                        # Found a cycle
                        # Cycle length = dist[u] + dist[v] + 1
                        # But this is only if u and v are in the same BFS tree?
                        # Actually, if we just want to detect ANY cycle < min_girth,
                        # we can check if dist[u] + dist[v] + 1 < min_girth.
                        # This works for undirected graphs.
                        if dist[u] + dist[v] + 1 < min_girth:
                            return False
            # If we finished BFS from start without finding short cycle, continue
        return True

    def generate_random_3_regular(n):
        # Configuration model
        # n must be even for 3-regular? No, sum of degrees = 3n must be even, so n must be even.
        if n % 2 != 0:
            return None
            
        half_edges = []
        for i in range(n):
            half_edges.extend([i, i, i])
        
        random.shuffle(half_edges)
        
        adj = [[] for _ in range(n)]
        edges = set()
        
        for i in range(0, len(half_edges), 2):
            u = half_edges[i]
            v = half_edges[i+1]
            
            if u == v:
                return None # Loop
            if (u, v) in edges or (v, u) in edges:
                return None # Multi-edge
            
            adj[u].append(v)
            adj[v].append(u)
            edges.add((min(u,v), max(u,v)))
            
        return n, list(edges), adj

    # Search
    best_n, best_edges = None, None
    
    # Start from 512, step 16 (to keep n even and increase probability)
    for n in range(512, 1000, 16):
        if time.time() > deadline:
            break
            
        # Try a few random instances for this n
        attempts = 0
        while attempts < 20 and time.time() < deadline:
            attempts += 1
            res = generate_random_3_regular(n)
            if res is None:
                continue
            
            n_res, edges, adj = res
            
            # Check girth
            if check_girth(n_res, adj, g):
                return n_res, edges
                
    # Fallback to LUW if no small graph found
    return luw_fallback()
