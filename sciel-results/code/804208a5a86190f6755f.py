def has_short_path(u, v, g, adj, n):
    if u == v: return False
    limit = g - 1
    if limit < 2: return False # girth >= 2
    
    visited = [False] * n
    dist = [0] * n
    queue = deque()
    queue.append(u)
    visited[u] = True
    
    while queue:
        curr = queue.popleft()
        curr_dist = dist[curr]
        if curr == v:
            return curr_dist < limit
        if curr_dist + 1 >= limit:
            continue
        for w in adj[curr]:
            if not visited[w]:
                visited[w] = True
                dist[w] = curr_dist + 1
                queue.append(w)
    return False
