import time
import numpy as np
from collections import deque

def construct():
    t0 = time.time()
    deadline = 80.0
    n = 16
    A = list(range(0, 8))
    B = list(range(8, 16))
    edges = [[a, b] for a in A for b in B]
    
    if _verify_fast(edges, n, deadline):
        return edges
    
    # fallback: star
    return [[0, v] for v in range(1, n)]

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
        maxd = max(maxd, max(dist))
    return maxd

def _bfs_diameter_matrix(adj_mat, n):
    """Compute diameter using BFS with numpy adjacency matrix. Returns None if disconnected."""
    maxd = 0
    for s in range(n):
        dist = np.full(n, -1, dtype=np.int32)
        dist[s] = 0
        q = deque([s])
        while q:
            u = q.popleft()
            neighbors = np.nonzero(adj_mat[u])[0]
            for w in neighbors:
                if dist[w] == -1:
                    dist[w] = dist[u] + 1
                    q.append(w)
        if np.any(dist == -1):
            return None
        maxd = max(maxd, int(np.max(dist)))
    return maxd

def _verify_fast(edges, n, deadline):
    """Verify diameter == 2 and every edge is critical, using numpy adjacency matrix."""
    adj_mat = np.zeros((n, n), dtype=np.int8)
    for u, v in edges:
        adj_mat[u, v] = 1
        adj_mat[v, u] = 1
    
    # Check diameter == 2
    d = _bfs_diameter_matrix(adj_mat, n)
    if d != 2:
        return False
    
    # Check each edge is critical
    for u, v in edges:
        if time.time() > deadline:
            # Time running out; assume remaining edges are critical (optimistic)
            # This is a heuristic - if we've verified most edges, likely all are critical
            pass
        
        # Temporarily remove edge (u,v)
        adj_mat[u, v] = 0
        adj_mat[v, u] = 0
        
        d2 = _bfs_diameter_matrix(adj_mat, n)
        if d2 is not None and d2 <= 2:
            # Edge is not critical - restore and fail
            adj_mat[u, v] = 1
            adj_mat[v, u] = 1
            return False
        
        # Restore edge
        adj_mat[u, v] = 1
        adj_mat[v, u] = 1
    
    return True
