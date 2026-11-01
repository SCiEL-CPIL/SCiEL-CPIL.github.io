"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct(k, g):
    import time
    start_time = time.time()
    deadline = start_time + 115  # 115s safety margin

    q = k
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add, sub, mul = (lambda x, y: x ^ y), (lambda x, y: x ^ y), (lambda x, y: table[x][y])
    else:
        add, sub, mul = (lambda x, y: (x + y) % q), (lambda x, y: (x - y) % q), (lambda x, y: x * y % q)

    def build_luw(d):
        """Build LUW(d) graph over GF(q). Returns (n, edges)."""
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

    def check_girth_bfs(n, edges, target_girth):
        """Check if girth >= target_girth using BFS from each vertex.
        For bipartite graphs, only even cycles matter.
        Returns True if girth >= target_girth."""
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        # Since the graph is bipartite, girth is even.
        # We need to check for cycles of length 2, 4, 6, ..., up to target_girth - 1 (or target_girth - 2 if even)
        # Max cycle length to check: target_girth - 1 if odd, target_girth - 2 if even
        # But since bipartite, we only check even lengths < target_girth
        
        max_depth = (target_girth - 1) // 2
        
        for start_node in range(n):
            # BFS from start_node
            dist = [-1] * n
            parent = [-1] * n
            dist[start_node] = 0
            queue = [start_node]
            qi = 0
            while qi < len(queue):
                node = queue[qi]
                qi += 1
                if dist[node] >= max_depth:
                    continue
                for neighbor in adj[node]:
                    if neighbor == parent[node]:
                        continue
                    if dist[neighbor] == -1:
                        dist[neighbor] = dist[node] + 1
                        parent[neighbor] = node
                        queue.append(neighbor)
                    else:
                        # Found a cycle. The cycle length is dist[node] + 1 + dist[neighbor]
                        # But this is only correct if the paths are disjoint except for start_node
                        # Actually, the cycle length is at least dist[node] + 1 + dist[neighbor]
                        # Wait, no. If we have an edge from node to neighbor, and both are in the BFS tree,
                        # the cycle goes from start_node to node (dist[node] steps), then node to neighbor (1 step),
                        # then neighbor back to start_node (dist[neighbor] steps).
                        # So cycle length = dist[node] + 1 + dist[neighbor]
                        cycle_len = dist[node] + 1 + dist[neighbor]
                        if cycle_len < target_girth:
                            return False
        return True

    # For LUW(d) over GF(q), the girth is at least 2d + 2
    # We need girth >= g
    # Since LUW is bipartite, girth is even, so we need girth >= g if g is even, girth >= g+1 if g is odd
    target_girth = g if g % 2 == 0 else g + 1
    
    # We need 2d + 2 >= target_girth, so d >= (target_girth - 2) / 2
    min_d = max(2, (target_girth - 2) // 2)
    
    # Try d values starting from min_d
    best_result = None
    for d in range(min_d, min_d + 8):
        if time.time() > deadline:
            break
        try:
            n, edges = build_luw(d)
            if check_girth_bfs(n, edges, target_girth):
                if best_result is None or n < best_result[0]:
                    best_result = (n, edges)
                return n, edges
        except Exception:
            continue
    
    # Fallback: just build with a reasonable d
    d = max(4, (target_girth - 2) // 2 + 2)
    n, edges = build_luw(d)
    return n, edges
# EVOLVE-BLOCK-END
