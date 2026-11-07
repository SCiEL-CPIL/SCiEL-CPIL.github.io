def construct():
    n = 15
    import time
    deadline = time.time() + 85  # safety margin before 90s kill

    # Start with K_{7,8}
    adj = [set() for _ in range(n)]
    deg = [0] * n
    edge_set = set()
    for u in range(7):
        for v in range(7, 15):
            adj[u].add(v)
            adj[v].add(u)
            deg[u] += 1
            deg[v] += 1
            edge_set.add((u, v))

    def is_valid():
        """Check if graph is valid: no 3-edge path has endpoints of equal degree."""
        for (a, b) in edge_set:
            for s in adj[a]:
                if s == b:
                    continue
                for t in adj[b]:
                    if t == a or t == s:
                        continue
                    if deg[s] == deg[t]:
                        return False
        return True

    def try_add_edge(u, v):
        """Try adding edge (u,v). Returns True if valid, False otherwise."""
        if u == v:
            return False
        key = (min(u, v), max(u, v))
        if key in edge_set:
            return False

        new_deg = deg[:]
        new_deg[u] += 1
        new_deg[v] += 1

        # Check new 3-edge paths using edge (u,v): s-u-v-t
        for s in adj[u]:
            if s == v:
                continue
            for t in adj[v]:
                if t == u or t == s:
                    continue
                if new_deg[s] == new_deg[t]:
                    return False

        # Check existing edges (a,b) where degree change at u or v creates violation
        # Only need to check edges incident to u or v
        for (a, b) in list(edge_set):
            if a == u or b == u or a == v or b == v:
                for s in adj[a]:
                    if s == b:
                        continue
                    for t in adj[b]:
                        if t == a or t == s:
                            continue
                        # Only check if s or t is u or v (degree changed)
                        if s == u or s == v or t == u or t == v:
                            if new_deg[s] == new_deg[t]:
                                return False
        return True

    def add_edge(u, v):
        """Add edge (u,v) to the graph."""
        key = (min(u, v), max(u, v))
        edge_set.add(key)
        adj[u].add(v)
        adj[v].add(u)
        deg[u] += 1
        deg[v] += 1

    def remove_edge(u, v):
        """Remove edge (u,v) from the graph."""
        key = (min(u, v), max(u, v))
        edge_set.discard(key)
        adj[u].discard(v)
        adj[v].discard(u)
        deg[u] -= 1
        deg[v] -= 1

    def greedy_fill():
        """Greedily add all possible edges."""
        changed = True
        while changed and time.time() < deadline - 5:
            changed = False
            for u in range(n):
                for v in range(u + 1, n):
                    if time.time() > deadline - 5:
                        return
                    key = (min(u, v), max(u, v))
                    if key in edge_set:
                        continue
                    if try_add_edge(u, v):
                        add_edge(u, v)
                        changed = True

    # Phase 1: Greedy edge addition
    greedy_fill()

    best_count = len(edge_set)
    best_state = (set(edge_set), [set(s) for s in adj], deg[:])

    # Phase 2: Edge swap local search
    # Try removing one edge and greedily refilling
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for (a, b) in list(edge_set):
            if time.time() > deadline:
                break
            # Save state
            orig_edge_set = set(edge_set)
            orig_adj = [set(s) for s in adj]
            orig_deg = deg[:]

            # Remove edge (a,b)
            remove_edge(a, b)

            # Try to add edges greedily
            greedy_fill()

            # Check if we improved
            if len(edge_set) > best_count:
                best_count = len(edge_set)
                best_state = (set(edge_set), [set(s) for s in adj], deg[:])
                improved = True
            else:
                # Restore
                edge_set.clear()
                edge_set.update(orig_edge_set)
                for i in range(n):
                    adj[i].clear()
                    adj[i].update(orig_adj[i])
                deg[:] = orig_deg

    # Restore best state
    if best_state is not None:
        edge_set.clear()
        edge_set.update(best_state[0])
        for i in range(n):
            adj[i].clear()
            adj[i].update(best_state[1][i])
        deg[:] = best_state[2]

    # Final validation
    if not is_valid():
        # Fallback: return K_{7,8}
        final_edges = []
        for u in range(7):
            for v in range(7, 15):
                final_edges.append([u, v])
        return final_edges

    # Build final edge list
    final_edges = []
    for (u, v) in edge_set:
        final_edges.append([u, v])

    return final_edges
