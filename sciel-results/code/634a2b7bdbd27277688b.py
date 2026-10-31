for nb in adj[node]:
    if nb == parent[node]:
        continue
    if dist[nb] == -1:
        dist[nb] = d + 1
        parent[nb] = node
        q.append(nb)
    else:
        L = d + dist[nb] + 1
        if L < min_girth:
            return False
