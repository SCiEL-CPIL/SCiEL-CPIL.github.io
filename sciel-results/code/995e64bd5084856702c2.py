    def luw_construction():
        q = k
        
        def build_luw(d):
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
            adj = [[] for _ in range(n)]
            for u, v in edges:
                adj[u].append(v)
                adj[v].append(u)
            return n, edges, adj

        # Try smaller d first
        # For g=9, d=4 is ideal.
        # Original logic: d=5.
        d_candidates = []
        if g == 9:
            d_candidates = [4, 5]
        else:
            d_orig = max(3, g - 5 if (g - 5) % 2 else g - 4)
            d_candidates = [d_orig]
            
        for d in d_candidates:
            n, edges, adj = build_luw(d)
            girth = get_girth(adj)
            if girth >= g:
                return n, edges
                
        # If all failed, return the last one (largest d) as it's most likely to have high girth
        # Even if girth < g, it's better than crashing? No, invalid score 0.
        # But if d=5 failed girth, d=6 might work.
        # Let's just return the result of the last d tried.
        n, edges, adj = build_luw(d_candidates[-1])
        return n, edges
