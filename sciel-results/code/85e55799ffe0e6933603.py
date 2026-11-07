def construct():
    n = 15
    import time
    deadline = time.time() + 85  # safety margin before 90s kill

    # Start with K_{7,8}
    edges = []
    for u in range(7):
        for v in range(7, 15):
            edges.append([u, v])

    # Build initial adjacency, degree, edge_set
    adj = [set() for _ in range(n)]
    deg = [0] * n
    edge_set = set()
    for (u, v) in edges:
        adj[u].add(v)
        adj[v].add(u)
        deg[u] += 1
        deg[v] += 1
        edge_set.add((min(u, v), max(u, v)))

    def is_valid(edge_set, adj, deg):
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

    def try_add_edge(edge_set, adj, deg, u, v):
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

    # Phase 1: Greedy edge addition
    changed = True
    while changed and time.time() < deadline - 20:
        changed = False
        for u in range(n):
            for v in range(u + 1, n):
                if time.time() > deadline - 20:
                    break
                key = (min(u, v), max(u, v))
                if key in edge_set:
                    continue
                if try_add_edge(edge_set, adj, deg, u, v):
                    edge_set.add(key)
                    adj[u].add(v)
                    adj[v].add(u)
                    deg[u] += 1
                    deg[v] += 1
                    changed = True
            if time.time() > deadline - 20:
                break

    # Phase 2: Edge swap local search (remove one edge, add another)
    best_edge_count = len(edge_set)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for (a, b) in list(edge_set):
            if time.time() > deadline:
                break
            # Temporarily remove edge (a,b)
            edge_set.discard((a, b))
            adj[a].discard(b)
            adj[b].discard(a)
            deg[a] -= 1
            deg[b] -= 1

            # Try to add a different edge
            for u in range(n):
                for v in range(u + 1, n):
                    if time.time() > deadline:
                        break
                    key = (min(u, v), max(u, v))
                    if key in edge_set:
                        continue
                    if try_add_edge(edge_set, adj, deg, u, v):
                        # Accept: we removed (a,b) and added (u,v), net same count
                        # But we want to try to add MORE edges after this
                        edge_set.add(key)
                        adj[u].add(v)
                        adj[v].add(u)
                        deg[u] += 1
                        deg[v] += 1
                        # Now try to add more edges
                        sub_changed = True
                        while sub_changed and time.time() < deadline:
                            sub_changed = False
                            for x in range(n):
                                for y in range(x + 1, n):
                                    if time.time() > deadline:
                                        break
                                    skey = (min(x, y), max(x, y))
                                    if skey in edge_set:
                                        continue
                                    if try_add_edge(edge_set, adj, deg, x, y):
                                        edge_set.add(skey)
                                        adj[x].add(y)
                                        adj[y].add(x)
                                        deg[x] += 1
                                        deg[y] += 1
                                        sub_changed = True
                                if time.time() > deadline:
                                    break
                        if len(edge_set) > best_edge_count:
                            best_edge_count = len(edge_set)
                            improved = True
                        # Don't break out of u,v loop; try more swaps
            # Restore original edge if we didn't accept a swap
            # Actually we need to track whether we accepted
            # Let me restructure this...
            break  # This approach is getting complicated; let me use a different strategy

    # Simpler Phase 2: Try removing each edge and see if we can add 2+ new edges
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
            edge_set.discard((a, b))
            adj[a].discard(b)
            adj[b].discard(a)
            deg[a] -= 1
            deg[b] -= 1

            # Try to add edges greedily
            sub_changed = True
            while sub_changed and time.time() < deadline:
                sub_changed = False
                for x in range(n):
                    for y in range(x + 1, n):
                        if time.time() > deadline:
                            break
                        skey = (min(x, y), max(x, y))
                        if skey in edge_set:
                            continue
                        if try_add_edge(edge_set, adj, deg, x, y):
                            edge_set.add(skey)
                            adj[x].add(y)
                            adj[y].add(x)
                            deg[x] += 1
                            deg[y] += 1
                            sub_changed = True
                    if time.time() > deadline:
                        break

            # Check if we improved
            if len(edge_set) > len(orig_edge_set):
                improved = True
                # Keep the new state
            else:
                # Restore
                edge_set = orig_edge_set
                adj = orig_adj
                deg = orig_deg

    # Final validation
    if not is_valid(edge_set, adj, deg):
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
