import time
import sys

def construct(k, g):
    start_time = time.time()
    # Safety margin: 10 seconds before 120s limit
    deadline = start_time + 110.0

    # Helper to check if graph has girth >= min_girth
    def check_girth(n, edges, min_girth):
        if n == 0:
            return True
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        # BFS from each vertex
        # To speed up, we can stop BFS if depth >= min_girth // 2 + 1?
        # No, we need to detect cycles of length < min_girth.
        # A cycle of length L is detected when an edge connects two nodes at depths d1, d2
        # such that d1 + d2 + 1 = L.
        # We only care if L < min_girth.
        # So we only need to explore depths up to min_girth // 2.
        # If no cycle found within this depth, girth >= min_girth.
        
        max_depth = min_girth // 2
        
        for s in range(n):
            dist = [-1] * n
            parent = [-1] * n
            dist[s] = 0
            queue = [(s, 0)]
            
            while queue:
                u, d = queue.pop(0)
                if d > max_depth:
                    continue
                
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = d + 1
                        parent[v] = u
                        queue.append((v, d + 1))
                    elif v != parent[u] and u != parent[v]:
                        # Found a cycle
                        cycle_len = dist[u] + dist[v] + 1
                        if cycle_len < min_girth:
                            return False
        
        return True

    def generate_luw(q, d):
        # Lazebnik-Ustimenko-Woldar graph CD(d, q)
        # q must be prime power
        if q == 4:
            table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
            add = lambda x, y: x ^ y
            sub = lambda x, y: x ^ y
            mul = lambda x, y: table[x][y]
        else:
            add = lambda x, y: (x + y) % q
            sub = lambda x, y: (x - y) % q
            mul = lambda x, y: x * y % q
            
        # Rule for coefficients
        rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
        for c in range(5, d + 1):
            r, c0 = (c - 5) % 4, c - (c - 5) % 4
            rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]

        # Vertices are (side, x) where x is tuple of length d+1 (x[0] is None)
        # Actually, x is a vector in GF(q)^d?
        # In original code: y = [None, first] + [0] * (d - 1)
        # So x has length d. x[0] is None, x[1..d-1] are values.
        # Wait, range(d-1) gives 0..d-2. So length is 1 + 1 + d-1 = d+1?
        # No, [None, first] is length 2. [0]*(d-1) is length d-1. Total d+1.
        # Indices 0..d.
        
        start = (0, (None,) + (0,) * d)
        index = {start: 0}
        order = [start]
        edges = []
        
        def neighbours(vertex):
            side, x = vertex
            for first in range(q):
                y = [None, first] + [0] * (d - 1)
                for c in range(2, d + 1):
                    a, b = rule[c]
                    # x and y are 0-indexed tuples/lists
                    # x[0] is None. x[1] is first component?
                    # In original: x[c] where c goes 2..d.
                    # So x has indices 0..d.
                    if side == 0:
                        # Point to Line
                        # l[c] - p[c] = l[a] * p[b]
                        # y[c] = x[c] + y[a] * x[b]
                        y[c] = add(x[c], mul(y[a], x[b]))
                    else:
                        # Line to Point
                        # p[c] = l[c] - l[a] * p[b] ?
                        # Original: sub(x[c], mul(x[a], y[b]))
                        y[c] = sub(x[c], mul(x[a], y[b]))
                yield 1 - side, tuple(y)

        for vertex in order:
            for other in neighbours(vertex):
                if other not in index:
                    index[other] = len(order)
                    order.append(other)
                if vertex[0] == 0:
                    edges.append((index[vertex], index[other]))
                    
        return len(order), edges

    # Determine candidate d values
    # Original logic for d
    original_d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    
    # We want to try smaller d first to minimize n
    # n ~ k^d
    # Try d from 3 to original_d
    best_n = float('inf')
    best_edges = None
    
    for d in range(3, original_d + 1):
        if time.time() > deadline:
            break
            
        n, edges = generate_luw(k, d)
        
        if n >= best_n:
            continue
            
        # Check girth
        # If this check is too slow, we might timeout.
        # But n is small for small d.
        
        if check_girth(n, edges, g):
            best_n = n
            best_edges = edges
            # Since n increases with d, the first valid d (smallest) is optimal
            # among these candidates.
            return best_n, best_edges
            
    # Fallback to original d if no smaller d worked or time ran out
    if best_edges is None:
        n, edges = generate_luw(k, original_d)
        return n, edges
        
    return best_n, best_edges
