"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct(k, g):
    import time
    start_time = time.time()
    deadline = start_time + 110  # 110s safety margin

    q = k
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add, sub, mul = (lambda x, y: x ^ y), (lambda x, y: x ^ y), (lambda x, y: table[x][y])
    else:
        add, sub, mul = (lambda x, y: (x + y) % q), (lambda x, y: (x - y) % q), (lambda x, y: x * y % q)

    def build_luw(d):
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

    def check_girth(n, edges, target_girth):
        # Check if girth >= target_girth
        # Since LUW is bipartite, girth is even.
        # We need to check for cycles of length < target_girth.
        # Max cycle length to check: target_girth - 1 (if odd) or target_girth - 2 (if even, since bipartite).
        # Actually, just check all even lengths < target_girth.
        # BFS from each vertex up to depth (target_girth - 1) // 2.
        
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
            
        max_depth = (target_girth - 1) // 2
        # For bipartite, we only care about even lengths.
        # If target_girth is 9, we check for cycles of length 2, 4, 6, 8.
        # Depth 4 covers length 8.
        
        for start_node in range(n):
            # BFS
            visited = {start_node: (None, 0)} # node: (parent, depth)
            queue = [(start_node, None, 0)]
            while queue:
                node, parent, depth = queue.pop(0)
                if depth == max_depth:
                    continue
                for neighbor in adj[node]:
                    if neighbor == parent:
                        continue
                    if neighbor in visited:
                        # Found a cycle
                        # Length = depth + 1 + visited[neighbor][1]
                        # But we need to be careful with the tree structure.
                        # A back-edge to a node at depth d' creates a cycle of length depth + 1 + d'?
                        # No, standard BFS cycle detection:
                        # If neighbor is visited and not parent, we have a cycle.
                        # The length is depth + 1 + visited[neighbor][1] IF they are in the same BFS tree?
                        # Actually, for girth checking, we just need to see if any cycle is shorter than target.
                        # If we find a back-edge, the cycle length is depth + 1 + dist(neighbor, start)?
                        # Simpler: Just run BFS up to max_depth. If we visit a node already visited, 
                        # and the path length is < target_girth, we have a short cycle.
                        
                        # The cycle length is depth + 1 + visited[neighbor][1] is not always correct 
                        # because the path to neighbor might not go through start in a way that forms a simple cycle with current path.
                        # However, for small depths, this is sufficient for a heuristic check.
                        # To be exact:
                        # If neighbor is visited, the cycle length is depth + 1 + visited[neighbor][1] 
                        # ONLY IF the LCA is start? No.
                        
                        # Correct check:
                        # If neighbor is visited, we have a cycle.
                        # The length is at least depth + 1 + visited[neighbor][1] ? No.
                        # Let's just check if depth + 1 + visited[neighbor][1] < target_girth.
                        # This is a lower bound on the cycle length?
                        # Actually, the cycle length is exactly depth + 1 + visited[neighbor][1] 
                        # if the paths are disjoint except for start.
                        # If they merge earlier, the cycle is shorter.
                        # So if depth + 1 + visited[neighbor][1] < target_girth, we definitely have a short cycle.
                        # If it's >= target_girth, we might still have a short cycle if they merge early.
                        # But for LUW graphs, the girth is large, so this check is likely sufficient.
                        
                        if depth + 1 + visited[neighbor][1] < target_girth:
                            return False
                    else:
                        visited[neighbor] = (node, depth + 1)
                        queue.append((neighbor, node, depth + 1))
        return True

    # Try smaller d first to get smaller n
    # d must be such that girth >= g
    # For odd g, we need girth >= g (which implies girth >= g+1 if bipartite? No, girth is even, so girth >= g means girth >= g if g even, girth >= g+1 if g odd?
    # "no cycle shorter than g".
    # If g=9, no cycles < 9. Bipartite graph has no odd cycles. So no cycles 1,3,5,7.
    # We must avoid cycles 2,4,6,8.
    # So we need girth >= 10? No, girth is the length of the shortest cycle.
    # If shortest cycle is 8, girth=8 < 9. Invalid.
    # If shortest cycle is 10, girth=10 >= 9. Valid.
    # So for bipartite graphs, "girth >= g" is equivalent to "girth >= g if g even, girth >= g+1 if g odd"?
    # No, girth is always even for bipartite.
    # So if g=9, we need girth >= 10?
    # Wait. "No cycle shorter than 9".
    # Cycles can be length 1, 2, ..., 8.
    # Bipartite has no 1, 3, 5, 7.
    # So we must avoid 2, 4, 6, 8.
    # So the shortest cycle must be >= 10?
    # If the shortest cycle is 8, it is shorter than 9. Invalid.
    # If the shortest cycle is 10, it is NOT shorter than 9. Valid.
    # So yes, for bipartite graphs, we need girth >= g if g is even, and girth >= g+1 if g is odd.
    
    target_girth_check = g if g % 2 == 0 else g + 1
    
    # Try d values
    # LUW girth bound: d+5 (odd d), d+4 (even d)?
    # We'll just try and check.
    
    candidates = []
    # Estimate d from g
    # If g=9, target check 10.
    # d=5 -> girth >= 10 (if bound is d+5).
    # d=4 -> girth >= 8 (if bound is d+4) or 9 (if d+5).
    # Let's try d=4 first.
    
    # Heuristic: d should be around g/2
    base_d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    
    # Try d in a range
    for d in [base_d - 1, base_d, base_d + 1]:
        if d < 2: continue
        if time.time() > deadline: break
        try:
            n, edges = build_luw(d)
            # Check girth
            if check_girth(n, edges, target_girth_check):
                return n, edges
        except Exception:
            continue
            
    # Fallback to original logic
    d = base_d
    n, edges = build_luw(d)
    return n, edges
# EVOLVE-BLOCK-END
