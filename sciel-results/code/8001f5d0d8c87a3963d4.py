<<<<<<< SEARCH
def _verify_girth(n, edges, g):
    """
    Verify the graph has girth >= g using BFS from each vertex.
    """
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    min_girth = float('inf')
    for start in range(n):
        dist = [-1] * n
        parent = [-1] * n
        dist[start] = 0
        queue = [start]
        
        for node in queue:
            for nb in adj[node]:
                if dist[nb] == -1:
                    dist[nb] = dist[node] + 1
                    parent[nb] = node
                    queue.append(nb)
                elif nb != parent[node]:
                    cycle_len = dist[node] + dist[nb] + 1
                    if cycle_len < min_girth:
                        min_girth = cycle_len
                        if min_girth < g:
                            return False
                            
    return min_girth >= g
=======
def _verify_girth(n, edges, g):
    """
    Verify the graph has girth >= g using BFS from each vertex.
    """
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    # Check regularity
    for i in range(n):
        if len(adj[i]) != len(set(adj[i])):
            return False  # Duplicate edges
    
    min_girth = float('inf')
    for start in range(n):
        dist = [-1] * n
        parent = [-1] * n
        dist[start] = 0
        queue = deque([start])
        
        while queue:
            node = queue.popleft()
            for nb in adj[node]:
                if dist[nb] == -1:
                    dist[nb] = dist[node] + 1
                    parent[nb] = node
                    queue.append(nb)
                elif nb != parent[node]:
                    cycle_len = dist[node] + dist[nb] + 1
                    if cycle_len < min_girth:
                        min_girth = cycle_len
                        if min_girth < g:
                            return False
                            
    return min_girth >= g
>>>>>>> REPLACE

