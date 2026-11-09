import time
from collections import deque

def _bfs_diameter(adj):
    """Return the diameter of the graph given by adjacency lists.
    Returns float('inf') if the graph is disconnected."""
    n = len(adj)
    diam = 0
    for start in range(n):
        dist = [-1] * n
        dist[start] = 0
        q = deque([start])
        while q:
            u = q.popleft()
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    q.append(v)
        for d in dist:
            if d == -1:
                return float('inf')
            if d > diam:
                diam = d
    return diam

def construct():
    start_time = time.time()
    n = 16
    half = n // 2
    edges = []
    
    # Construct K_{8,8}
    # This is the unique triangle-free graph with the maximum number of edges (64) on 16 vertices.
    # It has diameter 2.
    # Every edge is critical because the graph is triangle-free, so removing any edge (u,v)
    # means u and v have no common neighbor, forcing dist(u,v) >= 3 in the remaining graph.
    for u in range(half):
        for v in range(half, n):
            edges.append([u, v])
    
    # Safety deadline check
    if time.time() - start_time > 55:
        return edges
        
    # Return edges as list of [u, v] with Python ints
    result = []
    for u, v in edges:
        result.append([int(u), int(v)])
    return result
