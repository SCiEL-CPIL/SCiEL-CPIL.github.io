import time
import random

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 145.0
    
    # LUW construction
    def construct_luw(k, g, d_override=None):
        q = k
        if d_override is not None:
            d = d_override
        else:
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

    def check_girth(n, edges, g):
        # Build adjacency list
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        # Check for cycles of length < g using non-backtracking walks
        # BFS from each vertex, keeping track of parent to avoid backtracking
        # Since n is small (max ~3000), this is feasible.
        for start in range(n):
            # (node, parent, depth)
            queue = [(start, -1, 1)]
            visited = set()
            while queue:
                node, parent, depth = queue.pop(0)
                if depth >= g:
                    continue
                for neighbor in adj[node]:
                    if neighbor == parent:
                        continue
                    if neighbor == start and depth + 1 < g:
                        return False
                    if (neighbor, node) not in visited:
                        visited.add((neighbor, node))
                        queue.append((neighbor, node, depth + 1))
        return True

    # Try minimal d first
    d_min = (g - 1) // 2
    if d_min < 3:
        d_min = 3
        
    n_min, edges_min = construct_luw(k, g, d_min)
    if check_girth(n_min, edges_min, g):
        return n_min, edges_min
        
    # Fallback to original d
    n, edges = construct_luw(k, g)
    return n, edges
