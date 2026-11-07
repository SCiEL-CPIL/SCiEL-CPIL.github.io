    for u, v in added_edges:
        if adj[u][v]:
            return [] # Should not happen
        # Check triangle-free
        for w in range(n):
            if adj[u][w] and adj[v][w]:
                return [] # Invalid
