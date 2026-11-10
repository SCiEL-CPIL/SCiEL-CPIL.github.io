import time
from collections import deque

def construct():
    t0 = time.time()
    deadline = 85.0
    n = 16
    
    # K(8,8) complete bipartite: 64 edges
    A = list(range(0, 8))
    B = list(range(8, 16))
    edges = [[a, b] for a in A for b in B]
    
    if _verify_fast(edges, n):
        return edges
    
    # fallback: star (15 edges)
    return [[0, v] for v in range(1, n)]

def _verify_fast(edges, n):
    # Build adjacency sets
    adj = {v: set() for v in range(n)}
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
        
    # 1. Check Diameter is 2
    # Since it's K(8,8), we know it's 2, but let's verify generally.
    # Actually, for K(m,n), diameter is 2 if m,n >= 2.
    # We can skip BFS if we are confident, but let's do a quick BFS for safety.
    def has_diam_2():
        # Check connectivity and max dist
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
                return False
            if max(dist) > 2:
                return False
        return True

    if not has_diam_2():
        return False
        
    # 2. Check every edge is essential
    # Edge (u,v) is essential if removing it increases diameter > 2.
    # In a triangle-free graph, this is equivalent to u and v having no common neighbors.
    # If graph has triangles, this check is necessary but not sufficient?
    # If u,v have common neighbor w, d(u,v)=2 after removal.
    # But maybe other pairs become dist 3?
    # For K(8,8), it is triangle-free.
    # So we just check no common neighbors.
    for u, v in edges:
        if adj[u] & adj[v]:
            return False
            
    return True
