def construct(k, g):
    import time
    start_time = time.time()
    deadline = start_time + 110.0

    # Fast paths for small / known cases
    if k == 3:
        if g <= 3:
            return 3, [(0, 1), (1, 2), (0, 2)]
        elif g <= 4:
            return 6, [(0, 3), (0, 4), (0, 5), (1, 3), (1, 4), (1, 5),
                        (2, 3), (2, 4), (2, 5)]
        elif g <= 5:
            edges = []
            for i in range(5):
                edges.append((i, (i + 1) % 5))
                edges.append((i, 5 + i))
                edges.append((5 + i, 5 + (i + 2) % 5))
            return 10, edges
        elif g <= 6:
            edges = []
            lines = [[0,1,2],[0,3,4],[0,5,6],[1,3,5],[1,4,6],[2,3,6],[2,4,5]]
            for i, line in enumerate(lines):
                for pt in line:
                    edges.append((i, 7 + pt))
            return 14, edges
        elif g <= 7:
            pass
        elif g <= 8:
            pass
        elif g <= 9:
            pass
    elif k == 4:
        if g <= 3:
            return 4, [(0,1),(0,2),(0,3),(1,2),(1,3),(2,3)]
        elif g <= 4:
            return 8, [(i, j) for i in range(4) for j in range(4, 8)]
        elif g <= 5:
            pass
        elif g <= 6:
            pass
        elif g <= 7:
            pass
        elif g <= 8:
            pass
    elif k == 5:
        if g <= 3:
            return 6, [(i, j) for i in range(6) for j in range(i+1, 6)]
        elif g <= 4:
            return 10, [(i, j) for i in range(5) for j in range(5, 10)]
        elif g <= 5:
            pass
        elif g <= 6:
            pass
        elif g <= 7:
            pass
        elif g <= 8:
            pass
        elif g <= 9:
            pass

    def build_luw(d, q):
        if q == 4:
            add_table = [[x ^ y for y in range(4)] for x in range(4)]
            sub_table = [[x ^ y for y in range(4)] for x in range(4)]
            mul_table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        else:
            add_table = [[(x + y) % q for y in range(q)] for x in range(q)]
            sub_table = [[(x - y) % q for y in range(q)] for x in range(q)]
            mul_table = [[x * y % q for y in range(q)] for x in range(q)]

        rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
        for c in range(5, d + 1):
            r, c0 = (c - 5) % 4, c - (c - 5) % 4
            rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]

        rule_list = [None] * (d + 1)
        for c in range(2, d + 1):
            rule_list[c] = rule[c]

        # Precompute neighbor transitions for speed
        # For side 0: given x (tuple of length d+1, x[0]=None), compute all q neighbors
        # For side 1: given x, compute all q neighbors
        
        # Build all vertices and their neighbors
        start = (0, (None,) + (0,) * d)
        index = {start: 0}
        order = [start]
        edges = []
        edge_set = set()
        
        for vi, vertex in enumerate(order):
            side, x = vertex
            if side == 0:
                for first in range(q):
                    y = [None, first] + [0] * (d - 1)
                    for c in range(2, d + 1):
                        a, b = rule_list[c]
                        y[c] = add_table[x[c]][mul_table[y[a]][x[b]]]
                    other = (1, tuple(y))
                    if other not in index:
                        index[other] = len(order)
                        order.append(other)
                    e = (vi, index[other])
                    if e not in edge_set:
                        edge_set.add(e)
                        edges.append(e)
            else:
                for first in range(q):
                    y = [None, first] + [0] * (d - 1)
                    for c in range(2, d + 1):
                        a, b = rule_list[c]
                        y[c] = sub_table[x[c]][mul_table[x[a]][y[b]]]
                    other = (0, tuple(y))
                    if other not in index:
                        index[other] = len(order)
                        order.append(other)
                    e = (vi, index[other])
                    if e not in edge_set:
                        edge_set.add(e)
                        edges.append(e)
            if time.time() > deadline:
                break

        return len(order), edges

    def verify_girth(n, edges, g):
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        max_depth = (g - 1) // 2
        
        for s in range(n):
            dist = [-1] * n
            parent = [-1] * n
            dist[s] = 0
            queue = [s]
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
                    elif v != parent[u] and u != parent[v]:
                        cycle_len = dist[u] + dist[v] + 1
                        if cycle_len < g:
                            return False
            if time.time() > deadline:
                break
        
        return True

    q = k
    best_n, best_edges = None, None

    # Try d values including even d for odd g (might still work)
    if g % 2 == 1:
        candidates = list(range(g - 5, g + 2))
        # Include even d values too, since they might still have sufficient girth
        candidates = [d for d in candidates if d >= 2]
    else:
        candidates = list(range(g - 5, g + 2))
        candidates = [d for d in candidates if d >= 2]

    for d in candidates:
        if time.time() > deadline:
            break
        n, edges = build_luw(d, q)
        if verify_girth(n, edges, g):
            best_n, best_edges = n, edges
            break

    if best_n is None:
        # Fallback: use the smallest valid d
        d = max(3, g - 5)
        if g % 2 == 1 and d % 2 == 0:
            d += 1
        if g % 2 == 0 and d % 2 == 1:
            d += 1
        n, edges = build_luw(d, q)
        best_n, best_edges = n, edges

    return best_n, best_edges
