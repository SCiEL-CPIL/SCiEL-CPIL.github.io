def check_girth_fast(n, edges, g):
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
        
    # Check for any cycle of length < g
    # For each edge (u, v), check if there is a path from u to v of length < g-1 that does not use (u,v).
    # This is expensive.
    
    # Alternative: 
    # If g is small, BFS from each node to depth g//2.
    # If a node is reached by two different parents in the BFS tree, and the sum of depths + 1 < g, cycle found.
    
    for start in range(n):
        dist = [-1] * n
        parent = [-1] * n
        dist[start] = 0
        queue = deque([start])
        
        while queue:
            u = queue.popleft()
            if dist[u] >= g // 2:
                continue
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    parent[v] = u
                    queue.append(v)
                elif parent[v] != u and parent[u] != v:
                    # Found a cycle
                    # Length is dist[u] + dist[v] + 1
                    # But we need to be careful with the exact length.
                    # If dist[u] + dist[v] + 1 < g, then girth < g.
                    if dist[u] + dist[v] + 1 < g:
                        return False
                        
    return True
