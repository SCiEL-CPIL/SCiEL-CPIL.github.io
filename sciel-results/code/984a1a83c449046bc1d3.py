import time
import random
import numpy as np

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # Fallback: LUW Construction
    def luw_construct(k, g):
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

    # Random Search with Girth Check
    def generate_random_regular(n, k):
        # Configuration model
        if n * k % 2 != 0:
            return None
        stubs = []
        for i in range(n):
            for _ in range(k):
                stubs.append(i)
        random.shuffle(stubs)
        edges = set()
        adj = {i: [] for i in range(n)}
        for i in range(0, n * k, 2):
            u, v = stubs[i], stubs[i+1]
            if u == v:
                return None
            if (u, v) in edges or (v, u) in edges:
                return None
            edges.add((u, v))
            adj[u].append(v)
            adj[v].append(u)
        return list(edges)

    def has_short_cycle(n, edges, g):
        # BFS from each node to find shortest cycle
        # For speed, use a simplified check
        # Build adjacency list
        adj = {i: [] for i in range(n)}
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        # Check for cycles of length < g
        # This is expensive, so we only do it for small n or limited attempts
        # For g=10, we check for cycles of length 3,4,6,8
        # A simple BFS up to depth g/2 - 1
        
        # Optimized: Check for triangles, 4-cycles, etc.
        # Here we use a standard BFS to find the girth
        # If time is tight, we might skip this and rely on LUW
        
        # For this implementation, we do a quick check
        # If n > 500, skip detailed check to save time?
        # No, we need to verify.
        
        # Use BFS from each node
        for start in range(n):
            dist = [-1] * n
            parent = [-1] * n
            dist[start] = 0
            queue = [start]
            found_cycle = False
            while queue:
                u = queue.pop(0)
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        parent[v] = u
                        queue.append(v)
                    elif v != parent[u]:
                        # Cycle found
                        cycle_len = dist[u] + dist[v] + 1
                        if cycle_len < g:
                            return True
            # If no cycle found from start, continue
        return False

    # Strategy:
    # 1. Try random search for N in range [161, 5120]
    # 2. If found, return it.
    # 3. Else, return LUW.
    
    # Given the difficulty, random search is unlikely to find N=320 quickly.
    # We will limit the random search to a few seconds.
    
    best_n, best_edges = luw_construct(k, g)
    
    # If we have time, try to find a smaller graph
    # Moore bound for k=4, g=10 is 161.
    # We try N from 161 to best_n
    # But this is too slow.
    # We will just return LUW to be safe, unless we implement a faster check.
    
    # For now, return LUW.
    return best_n, best_edges
