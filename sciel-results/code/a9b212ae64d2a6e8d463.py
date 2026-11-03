def _calc_girth_from_vertex(n, edges, start):
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
        
    dist = [-1] * n
    parent = [-1] * n
    dist[start] = 0
    queue = [start]
    min_girth = float('inf')
    
    while queue:
        u = queue.pop(0)
        for v in adj[u]:
            if dist[v] == -1:
                dist[v] = dist[u] + 1
                parent[v] = u
                queue.append(v)
            elif v != parent[u]:
                cycle_len = dist[u] + dist[v] + 1
                if cycle_len < min_girth:
                    min_girth = cycle_len
                    if min_girth < 8:
                        return min_girth
    return min_girth if min_girth != float('inf') else n
