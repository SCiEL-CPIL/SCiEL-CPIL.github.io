import time
from collections import deque

def construct():
    t0 = time.time()
    deadline = 85.0
    n = 16
    A = list(range(0, 8))
    B = list(range(8, 16))
    edges = [[int(a), int(b)] for a in A for b in B]
    
    if _verify_fast(edges, n):
        return edges
    
    # fallback: star
    return [[int(0), int(v)] for v in range(1, n)]

def _bfs_diameter(adj, n):
    """Compute diameter using BFS from each vertex. Returns None if disconnected."""
    maxd = 0
    for s in range(n):
        dist = [-1] * n
        dist[s] = 0
        q = deque([s])
        while q:
            u = q.popleft()
            for w in adj[u]:
                if dist[w] == -1:
                    dist[w] = dist[u] + 1
                    q.append(w)
        if -1 in dist:
            return None
        local_max = max(dist)
        if local_max > maxd:
            maxd = local_max
    return maxd

def _verify_fast(edges, n):
    """Verify diameter == 2 and every edge is critical."""
    # Build adjacency as lists for faster iteration
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    # Check diameter == 2
    d = _bfs_diameter(adj, n)
    if d != 2:
        return False
    
    # Check each edge is critical using a faster method:
    # For edge (u,v), after removal, check if u and v still have distance <= 2
    # In a diameter-2 graph, removing (u,v) makes it non-critical iff
    # there exists a path u-w-v with w != u,v (i.e., u and v share a common neighbor)
    # But we need to be careful: we need ALL-pairs diameter > 2, not just u-v.
    # Actually for K(8,8), removing any edge (u,v) where u in A, v in B:
    # u and v are now distance 3 (u-a-b-v for a in A\{u}, b in B\{v})
    # But other pairs remain distance <= 2.
    # The diameter becomes 3 (between u and v), so it's critical.
    
    # For general verification, we need full BFS per edge removal.
    # Use list-based adjacency with temporary removal.
    for u, v in edges:
        # Remove edge (u,v)
        adj[u].remove(v)
        adj[v].remove(u)
        
        d2 = _bfs_diameter(adj, n)
        if d2 is not None and d2 <= 2:
            # Edge is not critical - restore and fail
            adj[u].append(v)
            adj[v].append(u)
            return False
        
        # Restore edge
        adj[u].append(v)
        adj[v].append(u)
    
    return True
