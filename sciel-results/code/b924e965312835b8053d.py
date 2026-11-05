for u in range(n_vertices):
    for v in adj[u]:
        if u < v:
            edges.append((u, v))
