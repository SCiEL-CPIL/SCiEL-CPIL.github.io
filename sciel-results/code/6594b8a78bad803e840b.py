import time
from collections import deque

def construct(k, g):
    deadline = time.time() + 105.0
    
    # Moore bound lower bound on n
    # For 3-regular with girth g: n >= 2 * (k-1)^(g//2) for even g
    # For odd g: n >= k * (k-1)^(g//2)
    if g % 2 == 0:
        half = g // 2
        lower = 2 * (k - 1) ** half
    else:
        half = g // 2
        lower = k * (k - 1) ** half
    
    # Known small cage graphs for specific (k,g) pairs
    # These are hard to construct generically, so we use known constructions
    
    # Try a few known constructions:
    # 1. For k=3, g=6: K_{3,3} (n=6)
    # 2. For k=3, g=8: Wagner graph / Möbius ladder M_8 (n=8)  
    # 3. For k=3, g=10: Petersen graph (n=10)
    # 4. For k=3, g=12: (n=24)
    # 5. For k=3, g=14: (n=38)
    # 6. For k=3, g=16: (n=936)
    
    # For the general case, use a heuristic search approach
    # Try random 3-regular graphs and check girth
    
    import random
    import numpy as np
    
    best_n = None
    best_edges = None
    
    # If we can't find anything better than the LUW construction, return that
    # First, let's try the LUW construction as fallback
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
    index, order, luw_edges = {start: 0}, [start], []
    for vertex in order:
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                luw_edges.append((index[vertex], index[other]))
    luw_n = len(order)
    best_n = luw_n
    best_edges = luw_edges
    
    # Now try to find something better with random search
    # For k=3, g=16, try to construct a graph with fewer than 936 vertices
    
    # Use a simple approach: try random cubic graphs and check girth
    # This is unlikely to work for large girth, but let's try with time limit
    
    def check_girth(edges, n, min_girth):
        """Check if graph has girth at least min_girth using BFS from each vertex."""
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        for start in range(n):
            dist = [-1] * n
            dist[start] = 0
            queue = deque([start])
            found = False
            while queue:
                u = queue.popleft()
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        if dist[v] >= min_girth:
                            continue
                        if dist[v] > min_girth // 2:
                            # Found a cycle of length <= min_girth
                            # Actually need to be more careful
                            pass
                        queue.append(v)
                    else:
                        # Back edge - check if this creates a short cycle
                        if dist[u] + 1 > min_girth:
                            continue
                        cycle_len = dist[u] + dist[v] + 1
                        if cycle_len < min_girth:
                            return False
            # Check for cycles through start
            for v in adj[start]:
                # Any edge from start to a neighbor at distance 0 would be a loop (not allowed)
                # Edges between neighbors of start create cycles of length 2 (not allowed in simple graph)
                pass
        return True
    
    # Faster girth check using BFS from each vertex
    def compute_girth(edges, n):
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        min_girth = float('inf')
        for start in range(n):
            dist = [-1] * n
            parent = [-1] * n
            dist[start] = 0
            queue = deque([start])
            while queue:
                u = queue.popleft()
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        parent[v] = u
                        queue.append(v)
                    elif v != parent[u] and u != parent[v]:
                        cycle_len = dist[u] + dist[v] + 1
                        if cycle_len < min_girth:
                            min_girth = cycle_len
        return min_girth
    
    # Try to construct a graph using known methods
    # For now, just return the LUW construction which is valid
    
    return best_n, best_edges
