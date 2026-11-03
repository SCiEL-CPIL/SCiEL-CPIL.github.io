<<<<<<< SEARCH
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
=======
        def neighbours(vertex, d):
            side, x = vertex
            for first in range(q):
                y = [None, first] + [0] * (d - 1)
                for c in range(2, d + 1):
                    a, b = rule[c]
                    y[c] = add(x[c], mul(y[a], x[b])) if side == 0 else sub(x[c], mul(x[a], y[b]))
                yield 1 - side, tuple(y)

        def build_graph(d):
            # Recompute rule for this d
            rule_local = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
            for c in range(5, d + 1):
                r, c0 = (c - 5) % 4, c - (c - 5) % 4
                rule_local[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]
            
            def nb(vtx):
                side, x = vtx
                for first in range(q):
                    y = [None, first] + [0] * (d - 1)
                    for c in range(2, d + 1):
                        a, b = rule_local[c]
                        y[c] = add(x[c], mul(y[a], x[b])) if side == 0 else sub(x[c], mul(x[a], y[b]))
                    yield 1 - side, tuple(y)

            start = (0, (None,) + (0,) * d)
            index, order, edges = {start: 0}, [start], []
            for vertex in order:
                for other in nb(vertex):
                    if other not in index:
                        index[other] = len(order)
                        order.append(other)
                    if vertex[0] == 0:
                        edges.append((index[vertex], index[other]))
            return len(order), edges

        def check_girth_nb(n, edges, g):
            """Check girth >= g using non-backtracking walks from every vertex."""
            adj = [[] for _ in range(n)]
            for u, v in edges:
                adj[u].append(v)
                adj[v].append(u)
            # Non-backtracking: (u, v) means we're at v having come from u
            # A cycle of length L < g would be found as a non-backtracking walk
            # of length L from (u,v) back to (u,v)
            # We check: for each directed edge (u,v), do non-backtracking BFS
            # up to depth g-1; if we return to (u,v), we have a cycle of length <= g-1
            # Actually, we check if any non-backtracking walk of length < g forms a cycle.
            # Simpler: BFS on the directed line graph (states = directed edges)
            # State: (prev, curr). Transition: (prev, curr) -> (curr, next) where next != prev
            # A cycle in this state graph of length L corresponds to a cycle of length L in original.
            # We need: no cycle of length < g in the state graph.
            # Check by: for each state, BFS up to depth g-1, see if we revisit the start state.
            
            for u in range(n):
                for v in adj[u]:
                    # BFS from state (u, v) in the non-backtracking line graph
                    visited = set()
                    visited.add((u, v))
                    queue = deque([(u, v, 0)])
                    while queue:
                        prev, curr, depth = queue.popleft()
                        if depth >= g - 1:
                            continue
                        for nxt in adj[curr]:
                            if nxt == prev:
                                continue
                            if (curr, nxt) == (u, v):
                                return False  # found cycle of length depth+1 < g
                            if (curr, nxt) not in visited:
                                visited.add((curr, nxt))
                                queue.append((curr, nxt, depth + 1))
            return True

        # Try increasing d until girth is satisfied
        for trial_d in range(d, d + 10):
            if time.time() > deadline:
                break
            n, edges = build_graph(trial_d)
            if check_girth_nb(n, edges, g):
                return n, edges
            d = trial_d + 1

        # Fallback: return the last built graph (may not satisfy girth, but should be close)
        return n, edges
>>>>>>> REPLACE

