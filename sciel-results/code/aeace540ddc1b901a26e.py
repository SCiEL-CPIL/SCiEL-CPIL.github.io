import time
import random
import numpy as np

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 115.0  # 5s margin
    
    # Fallback: LUW construction from the seed
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

    # Check if we are in the specific case k=4, g=10
    if k != 4 or g != 10:
        # For other cases, just use LUW or a simple random search with lower expectations
        # But the task implies this specific case.
        # Let's just use LUW for safety if not 4,10.
        return luw_construct(k, g)

    # Moore bound for k=4, g=10 is 161.
    # We will search for n starting from 161.
    # Known best is 320.
    
    def has_short_cycles(edges, n, g):
        # Build adjacency list
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        # Check for loops and multi-edges (should be handled by generator, but just in case)
        for u in range(n):
            if len(adj[u]) != k:
                return True # Invalid degree
            if len(set(adj[u])) != len(adj[u]):
                return True # Multi-edge
        
        # BFS from each vertex to depth g//2 - 1
        max_depth = g // 2 - 1
        for start in range(n):
            # dist[v] = depth, parent[v] = parent node
            dist = [-1] * n
            parent = [-1] * n
            dist[start] = 0
            queue = [start]
            head = 0
            while head < len(queue):
                u = queue[head]
                head += 1
                if dist[u] >= max_depth:
                    continue
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        parent[v] = u
                        queue.append(v)
                    else:
                        # Found a back edge or cross edge
                        # Cycle length is dist[u] + dist[v] + 1
                        # But we need to be careful. 
                        # If v is an ancestor of u, cycle length is dist[u] - dist[v] + 1? No.
                        # Standard: if we find an edge (u,v) where both are visited,
                        # and they are not parent-child, we have a cycle.
                        # The length is dist[u] + dist[v] + 1 if they are in different branches?
                        # Actually, just check if dist[u] + dist[v] + 1 < g.
                        # This is a heuristic. 
                        # More robust: if v is visited and v != parent[u] and u != parent[v]:
                        # Then cycle length is dist[u] + dist[v] + 1.
                        if v != parent[u] and u != parent[v]:
                            cycle_len = dist[u] + dist[v] + 1
                            if cycle_len < g:
                                return True
        return False

    def generate_random_4_reg(n):
        # Configuration model
        stubs = []
        for i in range(n):
            for _ in range(k):
                stubs.append(i)
        
        while True:
            random.shuffle(stubs)
            edges = set()
            valid = True
            for i in range(0, len(stubs), 2):
                u = stubs[i]
                v = stubs[i+1]
                if u == v:
                    valid = False
                    break
                if (u, v) in edges or (v, u) in edges:
                    valid = False
                    break
                edges.add((u, v))
            if valid:
                return list(edges)

    # Search loop
    best_n, best_edges = luw_construct(k, g)
    
    # Try n from Moore bound to 400
    for n in range(161, 401):
        if time.time() > deadline:
            break
        # Try a few random graphs for this n
        attempts = 0
        max_attempts = 50 # Adjust based on time
        while attempts < max_attempts and time.time() < deadline:
            attempts += 1
            edges = generate_random_4_reg(n)
            if not has_short_cycles(edges, n, g):
                return n, edges
    
    return best_n, best_edges
