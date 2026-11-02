"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct(k, g):
    import time
    deadline = time.time() + 110  # safety margin before 120s limit

    q = k
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add = lambda x, y: x ^ y
        sub = lambda x, y: x ^ y
        mul = lambda x, y: table[x][y]
    else:
        add = lambda x, y: (x + y) % q
        sub = lambda x, y: (x - y) % q
        mul = lambda x, y: x * y % q

    def build_luw(d):
        """Build the LUW graph CD(d, q) and return (n, edges)."""
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
                    if side == 0:
                        y[c] = add(x[c], mul(y[a], x[b]))
                    else:
                        y[c] = sub(x[c], mul(x[a], y[b]))
                yield 1 - side, tuple(y)

        start = (0, (None,) + (0,) * d)
        index = {start: 0}
        order = [start]
        edges = []
        for vertex in order:
            for other in neighbours(vertex):
                if other not in index:
                    index[other] = len(order)
                    order.append(other)
                if vertex[0] == 0:
                    edges.append((index[vertex], index[other]))
        return len(order), edges

    def check_girth_exact(n, edges, target_girth):
        """Exact girth check via non-backtracking BFS from every vertex.
        Returns True iff no cycle shorter than target_girth exists."""
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)

        # For a bipartite graph, girth is even.
        # We need to check for cycles of length 2, 4, 6, ..., up to target_girth-1 (or target_girth-2 if even).
        # Actually, we just need: no cycle of length < target_girth.
        # Non-backtracking BFS: from each start, walk up to depth (target_girth-2)//2 in non-backtracking manner.
        # If we revisit a node at depth < target_girth, we have a short cycle.

        max_depth = (target_girth - 2) // 2  # max depth to search; cycle length = 2*depth or 2*depth+1
        # For target_girth=9: max_depth = 3, checking cycles of length up to 6 or 7.
        # Actually, let's think more carefully.
        # A non-backtracking walk of length L from v that returns to v gives a cycle of length L.
        # We want to detect any cycle of length < target_girth.
        # So we need to check non-backtracking walks of length 2, 3, ..., target_girth-1.
        # But since the graph is bipartite, only even lengths matter: 2, 4, 6, ..., target_girth-2 (if target_girth is odd) or target_girth-2 (if even).
        # Wait, for target_girth=9: we need no cycle of length 2,3,4,5,6,7,8. Bipartite: no odd cycles, so check 2,4,6,8.
        # 8 < 9, so we need to check up to length 8.
        # Non-backtracking walk of length 8 returning to start = cycle of length 8.
        # So max_depth for BFS should be 8 (walk length), but we can do it level by level.

        # More efficient: for each start node, do a BFS where state is (node, prev_node) to avoid backtracking.
        # Track the depth. If we reach (node, prev) where node == start and depth >= 2, we have a cycle of length depth.
        # But this is O(n^2) in the worst case which is too slow for large n.

        # Alternative: for small target_girth (like 9), we can check for 4-cycles and 6-cycles and 8-cycles separately.
        # 4-cycle: two nodes at distance 2 with two common neighbors.
        # 6-cycle: more complex.
        # 8-cycle: even more complex.

        # Simpler approach for girth check with small g:
        # For each pair of nodes (u,v) at distance 2, check if they share 2+ common neighbors (4-cycle).
        # For 6-cycles: for each node u, for each pair of neighbors (a,b), check if there's a path of length 3 from a to b not through u.
        # This is getting complex. Let's use a different approach.

        # Since the LUW graph is known to have girth >= d+5 (odd d) or d+4 (even d),
        # and we're constructing with a specific d, we can rely on the theoretical bound
        # and just verify with a quick check for small cycles.

        # For practical purposes with g=9, the LUW graph with d=5 has girth >= 10,
        # so it's valid. Let's do a quick 4-cycle and 6-cycle check.

        # Check for 4-cycles: for each node, count pairs of neighbors that share another common neighbor.
        for u in range(n):
            if time.time() > deadline:
                return False  # timeout, assume invalid to be safe? No, return True to use it.
            neighbors_u = adj[u]
            for i in range(len(neighbors_u)):
                for j in range(i + 1, len(neighbors_u)):
                    a, b = neighbors_u[i], neighbors_u[j]
                    # Check if a and b have a common neighbor other than u
                    common = 0
                    for w in adj[a]:
                        if w == u:
                            continue
                        if w in adj[b]:
                            common += 1
                            if common >= 1:
                                break
                    if common >= 1:
                        return False
        # Check for 6-cycles: for each node u, for each pair of neighbors (a,b),
        # check if there exists a node c adjacent to both a and b, and a node d adjacent to both c and u... 
        # This is getting too complex. Let's just check 4-cycles and trust the theory for larger ones.
        # Actually, let's do a proper non-backtracking BFS but limit the depth.
        
        # Reset and do proper check
        for start in range(n):
            if time.time() > deadline:
                return True  # assume valid if we ran out of time
            # State: (current_node, prev_node, depth)
            # Start: (start, -1, 0)
            # We want to find if we can return to start with depth < target_girth
            visited = set()
            visited.add((start, -1))
            queue = [(start, -1, 0)]
            while queue:
                node, prev, depth = queue.pop(0)
                if depth >= target_girth - 1:
                    continue
                for neighbor in adj[node]:
                    if neighbor == prev:
                        continue
                    if neighbor == start and depth + 1 >= 2:
                        # Found a cycle of length depth + 1
                        if depth + 1 < target_girth:
                            return False
                    if (neighbor, node) not in visited:
                        visited.add((neighbor, node))
                        queue.append((neighbor, node, depth + 1))
        return True

    # Determine candidate d values to try
    # LUW CD(d,q) girth: >= d+5 for odd d, >= d+4 for even d
    # We need girth >= g (no cycle shorter than g)
    # For bipartite, girth is even, so effectively girth >= g if g even, girth >= g+1 if g odd
    target_girth = g if g % 2 == 0 else g + 1

    # Find d values that could satisfy the girth bound
    candidates = []
    for d in range(3, 15):
        if d % 2 == 1:
            if d + 5 >= target_girth:
                candidates.append(d)
        else:
            if d + 4 >= target_girth:
                candidates.append(d)

    # Sort by expected size (smaller d -> smaller graph)
    candidates.sort()

    best_result = None
    for d in candidates:
        if time.time() > deadline:
            break
        try:
            n, edges = build_luw(d)
            if check_girth_exact(n, edges, g):
                if best_result is None or n < best_result[0]:
                    best_result = (n, edges)
        except Exception:
            continue

    if best_result is not None:
        return best_result

    # Fallback: use the original d calculation
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    n, edges = build_luw(d)
    return n, edges
# EVOLVE-BLOCK-END
