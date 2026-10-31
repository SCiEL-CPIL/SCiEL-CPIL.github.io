def _extract_component(n, edges, start=0):
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    visited = set([start])
    stack = [start]
    while stack:
        u = stack.pop()
        for v in adj[u]:
            if v not in visited:
                visited.add(v)
                stack.append(v)
                
    if len(visited) == n:
        return n, edges
        
    # Re-index
    old_to_new = {v: i for i, v in enumerate(sorted(visited))}
    new_edges = []
    for u, v in edges:
        if u in old_to_new and v in old_to_new:
            new_u = old_to_new[u]
            new_v = old_to_new[v]
            if new_u < new_v:
                new_edges.append((new_u, new_v))
            
    return len(visited), new_edges
