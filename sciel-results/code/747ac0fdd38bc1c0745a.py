import time
import random
import numpy as np

def construct(k, g):
    # Deadline guard
    start_time = time.time()
    deadline = 140  # 10s margin
    
    # Try to find a smaller graph using a modified LUW construction with smaller d
    # The LUW graph CD(d, q) has girth at least d+5.
    # For g=16, we need d+5 >= 16 => d >= 11.
    # However, if d is even, the girth might be d+6.
    # Let's test d=10. If girth >= 16, use it.
    
    q = k
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add, sub, mul = (lambda x, y: x ^ y), (lambda x, y: x ^ y), (lambda x, y: table[x][y])
    else:
        add, sub, mul = (lambda x, y: (x + y) % q), (lambda x, y: (x - y) % q), (lambda x, y: x * y % q)
        
    def get_rule(d):
        rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
        for c in range(5, d + 1):
            r, c0 = (c - 5) % 4, c - (c - 5) % 4
            rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]
        return rule

    def build_luw(d):
        rule = get_rule(d)
        
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

    def check_girth(n, edges, required_girth):
        # Build adjacency list
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
            
        # Check girth by BFS from each vertex
        # To save time, we can just check a few vertices or use a heuristic
        # But for correctness, we must be sure.
        # Given the size (potentially 100k+), full BFS is slow.
        # We assume the algebraic construction is correct for d=10 if girth is d+6=16.
        # If d=10 gives girth 15, it's invalid.
        # We will do a quick check on a subset of vertices.
        
        # Actually, for d=10, the LUW graph is known to have girth 15 or 16?
        # The paper says girth is d+5. So d=10 -> girth 15.
        # So d=10 is likely invalid for g=16.
        
        # Therefore, we must use d=11.
        
        # However, we can try to **reduce the size** by using a **quotient** or **subgraph**?
        # No.
        
        # So we return the d=11 graph.
        return True # Placeholder, we will just build d=11

    # Build with d=11 (guaranteed girth >= 16)
    d = 11
    n, edges = build_luw(d)
    
    # Optional: Try to verify girth on a small sample to be safe, but it's known to be valid.
    
    return n, edges
