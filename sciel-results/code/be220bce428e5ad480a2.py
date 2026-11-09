"""Dense diameter-2 graph with every edge critical."""
# EVOLVE-BLOCK-START
import time

def _bfs(adj, src, n):
    dist = [-1] * n
    dist[src] = 0
    q = [src]
    qi = 0
    while qi < len(q):
        u = q[qi]; qi += 1
        du = dist[u] + 1
        for v in adj[u]:
            if dist[v] == -1:
                dist[v] = du
                q.append(v)
    return dist

def construct():
    start = time.time()
    deadline = start + 70.0
    n = 16
    
    # Try complement = perfect matching (8 disjoint edges)
    # Graph = K16 minus 8 edges = 112 edges
    # Check if valid
    adj = [[] for _ in range(n)]
    edges = []
    for u in range(n):
        for v in range(u+1, n):
            adj[u].append(v)
            adj[v].append(u)
            edges.append([u, v])
    # Remove matching edges: (0,1), (2,3), (4,5), (6,7), (8,9), (10,11), (12,13), (14,15)
    matching = [(0,1), (2,3), (4,5), (6,7), (8,9), (10,11), (12,13), (14,15)]
    for u, v in matching:
        adj[u].remove(v)
        adj[v].remove(u)
        edges.remove([u, v])
    
    # Check diameter
    valid = True
    for s in range(n):
        dist = _bfs(adj, s, n)
        if max(dist) != 2:
            valid = False
            break
    
    if valid:
        # Check criticality
        all_critical = True
        for edge in edges:
            u, v = edge
            adj[u].remove(v)
            adj[v].remove(u)
            ok = True
            for s in range(n):
                dist = _bfs(adj, s, n)
                if any(d == -1 for d in dist) or max(dist) > 2:
                    ok = False
                    break
            adj[u].append(v)
            adj[v].append(u)
            if ok:
                all_critical = False
                break
        if all_critical:
            return [[int(u), int(v)] for u, v in edges]
    
    # Fallback: try a known small construction
    # Use the seed's star as fallback
    return [[0, v] for v in range(1, n)]
# EVOLVE-BLOCK-END
