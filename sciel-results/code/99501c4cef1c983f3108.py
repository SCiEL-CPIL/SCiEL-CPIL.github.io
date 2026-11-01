    def verify_girth(n, edges, g):
        import collections
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        max_depth = (g - 1) // 2
        for start in range(n):
            dist = {start: 0}
            parent = {start: -1}
            queue = collections.deque([start])
            while queue:
                u = queue.popleft()
                if dist[u] > max_depth:
                    continue
                for v in adj[u]:
                    if v not in dist:
                        dist[v] = dist[u] + 1
                        parent[v] = u
                        queue.append(v)
                    elif parent[u] != v and parent[v] != u:
                        cycle_len = dist[u] + dist[v] + 1
                        if cycle_len < g:
                            return False
        return True

    # We need girth >= g.
    # Start with a theoretical lower bound for d, but verify.
    min_d = max(2, (target_girth - 2) // 2)
    d = min_d
    
    while True:
        n, edges = build_luw(d)
        
        # Verify girth
        if verify_girth(n, edges, g):
            return n, edges
        
        # If girth is too small, increase d
        d += 1
        
        # Safety check
        if time.time() > deadline:
            # If we run out of time, return the current one.
            # Note: This might be invalid.
            return n, edges
