def check_girth(edges, n, g):
    # Build adjacency list
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
        
    for start in range(n):
        # BFS
        dist = [-1] * n
        parent = [-1] * n
        dist[start] = 0
        queue = [start]
        
        while queue:
            u = queue.pop(0)
            if dist[u] >= (g - 1) // 2:
                continue
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    parent[v] = u
                    queue.append(v)
                elif v != parent[u]:
                    # Found a cycle
                    # Cycle length is dist[u] + dist[v] + 1
                    # But this is not necessarily the shortest cycle through start
                    # However, if dist[u] + dist[v] + 1 < g, we have a short cycle
                    if dist[u] + dist[v] + 1 < g:
                        return False
    return True
