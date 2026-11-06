        if len(adj[u]) < len(adj[v]):
            common_count = sum(1 for x in adj[u] if x in adj[v])
        else:
            common_count = sum(1 for x in adj[v] if x in adj[u])
        if common_count <= 1:
