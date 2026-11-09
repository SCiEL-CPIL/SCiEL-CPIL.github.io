<<<<<<< SEARCH
    # Binary search
    while lo <= hi and time.time() < deadline:
        mid = (lo + hi) // 2
        
        # Build subgraph with edges at position <= mid
        adj = [[] for _ in range(n)]
        for (u, v), pos in edge_pos.items():
            if pos <= mid:
                adj[u].append(v)
                adj[v].append(u)
        
        # Check for Hamilton cycle
        cycle = find_cycle_in_subgraph(adj, n)
        
        if cycle is not None:
            best_cycle = cycle
            best_score = mid
            hi = mid - 1
        else:
            lo = mid + 1
=======
    # Binary search
    while lo <= hi and time.time() < deadline:
        mid = (lo + hi) // 2
        
        # Build subgraph with edges at position <= mid
        adj = [[] for _ in range(n)]
        for (u, v), pos in edge_pos.items():
            if pos <= mid:
                adj[u].append(v)
                adj[v].append(u)
        
        # Quick feasibility check before expensive backtracking
        feasible = True
        for v in range(n):
            if len(adj[v]) < 2:
                feasible = False
                break
        if feasible:
            # Check connectivity
            visited = [False] * n
            stack = [0]
            visited[0] = True
            cnt = 0
            while stack:
                u = stack.pop()
                cnt += 1
                for w in adj[u]:
                    if not visited[w]:
                        visited[w] = True
                        stack.append(w)
            if cnt != n:
                feasible = False
        
        if not feasible:
            lo = mid + 1
            continue
        
        # Check for Hamilton cycle
        cycle = find_cycle_in_subgraph(adj, n)
        
        if cycle is not None:
            best_cycle = cycle
            best_score = mid
            hi = mid - 1
        else:
            lo = mid + 1
>>>>>>> REPLACE

