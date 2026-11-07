def _is_valid_fast(graph, signs):
    """Check if lift is connected (i.e., signs are not a coboundary)."""
    n = graph['n']
    edges = graph['edges']
    
    # Build adjacency list with signs
    adj = [[] for _ in range(n)]
    for s, (u, v) in zip(signs, edges):
        adj[u].append((v, s))
        adj[v].append((u, s))
        
    visited = [-1] * n
    visited[0] = 1  # sigma_0 = 1
    
    stack = [0]
    while stack:
        u = stack.pop()
        for v, s in adj[u]:
            expected_sigma = visited[u] * s
            if visited[v] == -1:
                visited[v] = expected_sigma
                stack.append(v)
            else:
                if visited[v] != expected_sigma:
                    # Conflict found, not a coboundary -> Valid
                    return True
                    
    # No conflict, it is a coboundary -> Invalid
    return False
