<<<<<<< SEARCH
def _check_girth_bfs(n, edges, min_girth):
    """
    Checks if the graph has girth at least min_girth.
    Returns True if girth >= min_girth.
    """
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
        
    max_depth = min_girth // 2 - 1
    
    for start in range(n):
        dist = [-1] * n
        parent = [-1] * n
        dist[start] = 0
        queue = collections.deque([start])
        
        while queue:
            curr = queue.popleft()
            if dist[curr] > max_depth:
                continue
            for neighbor in adj[curr]:
                if dist[neighbor] == -1:
                    dist[neighbor] = dist[curr] + 1
                    parent[neighbor] = curr
                    queue.append(neighbor)
                elif neighbor != parent[curr]:
                    # Found a cycle
                    if dist[curr] + dist[neighbor] + 1 < min_girth:
                        return False
    return True
=======
def _check_girth_bfs(n, edges, min_girth):
    """
    Checks if the graph has girth at least min_girth.
    Returns True if girth >= min_girth.
    Uses adjacency lists for speed.
    """
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
        
    max_depth = min_girth // 2 - 1
    if max_depth < 1:
        return True
    
    for start in range(n):
        dist = [-1] * n
        parent = [-1] * n
        dist[start] = 0
        queue = collections.deque([start])
        
        while queue:
            curr = queue.popleft()
            d = dist[curr]
            if d > max_depth:
                continue
            for neighbor in adj[curr]:
                if dist[neighbor] == -1:
                    dist[neighbor] = d + 1
                    parent[neighbor] = curr
                    queue.append(neighbor)
                elif neighbor != parent[curr]:
                    if d + dist[neighbor] + 1 < min_girth:
                        return False
    return True
>>>>>>> REPLACE

