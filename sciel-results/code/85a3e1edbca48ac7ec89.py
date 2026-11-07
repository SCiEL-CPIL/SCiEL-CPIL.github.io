def construct():
    n = 15
    import time
    deadline = time.time() + 85

    # Start with K_{7,8}
    edge_set = set()
    adj = [set() for _ in range(n)]
    deg = [0] * n
    for u in range(7):
        for v in range(7, 15):
            edge_set.add((u, v))
            adj[u].add(v)
            adj[v].add(u)
            deg[u] += 1
            deg[v] += 1

    def is_valid():
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

    def try_add(u, v):
        """Try adding edge (u,v). Returns True if valid after addition."""
        if u == v:
            return False
        key = (min(u, v), max(u, v))
        if key in edge_set:
            return False

        # Check new paths using edge (u,v): s-u-v-t
        for s in adj[u]:
            if s == v:
                continue
            for t in adj[v]:
                if t == u or t == s:
                    continue
                if (deg[s] + (1 if s == u else 0) + (1 if s == v else 0)) == \
                   (deg[t] + (1 if t == u else 0) + (1 if t == v else 0)):
                    return False

        # Check existing edges (a,b) where a or b is u or v
        # Only paths where s or t is u or v can be affected (degree changes)
        for (a, b) in edge_set:
            if a == u or b == u or a == v or b == v:
                for s in adj[a]:
                    if s == b:
                        continue
                    for t in adj[b]:
                        if t == a or t == s:
                            continue
                        if s == u or s == v or t == u or t == v:
                            ns = deg[s] + (1 if s == u else 0) + (1 if s == v else 0)
                            nt = deg[t] + (1 if t == u else 0) + (1 if t == v else 0)
                            if ns == nt:
                                return False
        return True

    def do_add(u, v):
        key = (min(u, v), max(u, v))
        edge_set.add(key)
        adj[u].add(v)
        adj[v].add(u)
        deg[u] += 1
        deg[v] += 1

    def do_remove(u, v):
        key = (min(u, v), max(u, v))
        edge_set.discard(key)
        adj[u].discard(v)
        adj[v].discard(u)
        deg[u] -= 1
        deg[v] -= 1

    # Phase 1: Greedy edge addition - try all missing edges
    improved = True
    while improved and time.time() < deadline - 30:
        improved = False
        for u in range(n):
            for v in range(u + 1, n):
                if time.time() > deadline - 30:
                    break
                key = (min(u, v), max(u, v))
                if key in edge_set:
                    continue
                if try_add(u, v):
                    do_add(u, v)
                    improved = True
            if time.time() > deadline - 30:
                break

    # Phase 2: Edge swap local search - remove one edge, try to add two
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for (a, b) in list(edge_set):
            if time.time() > deadline:
                break
            # Save state
            saved_edges = set(edge_set)
            saved_adj = [set(s) for s in adj]
            saved_deg = deg[:]

            # Remove edge (a,b)
            do_remove(a, b)

            # Try to add edges greedily
            sub_improved = True
            while sub_improved and time.time() < deadline:
                sub_improved = False
                for u in range(n):
                    for v in range(u + 1, n):
                        if time.time() > deadline:
                            break
                        key = (min(u, v), max(u, v))
                        if key in edge_set:
                            continue
                        if try_add(u, v):
                            do_add(u, v)
                            sub_improved = True
                    if time.time() > deadline:
                        break

            # Check if we improved
            if len(edge_set) > len(saved_edges):
                improved = True
                # Keep new state
            else:
                # Restore
                edge_set = saved_edges
                adj = saved_adj
                deg = saved_deg

    # Final validation
    if not is_valid():
        # Fallback: return K_{7,8}
        final_edges = []
        for u in range(7):
            for v in range(7, 15):
                final_edges.append([int(u), int(v)])
        return final_edges

    # Build final edge list
    final_edges = []
    for (u, v) in edge_set:
        final_edges.append([int(u), int(v)])

    return final_edges
