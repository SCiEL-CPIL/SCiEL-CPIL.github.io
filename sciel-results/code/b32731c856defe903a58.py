"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
from collections import deque

P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}

def _build_adj(edges, n):
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    return adj

def _bfs_distances(adj, src, n):
    dist = [-1] * n
    dist[src] = 0
    q = deque([src])
    while q:
        u = q.popleft()
        for v in adj[u]:
            if dist[v] == -1:
                dist[v] = dist[u] + 1
                q.append(v)
    return dist

def _max_diameter(adj, n):
    maxd = 0
    for s in range(n):
        d = _bfs_distances(adj, s, n)
        maxd = max(maxd, max(d))
    return maxd

def _is_triangle_free(adj, u, v, n):
    # Check if adding edge (u,v) creates a triangle
    # Triangle exists if there's a common neighbor of u and v
    nu = set(adj[u])
    for w in adj[v]:
        if w in nu:
            return False
    return True

def construct():
    n = P['n']
    base_edges = [tuple(e) for e in P['edges']]
    edge_set = set(base_edges)
    # Normalize: store as frozenset of tuples
    norm_edges = set()
    for u, v in base_edges:
        norm_edges.add((min(u,v), max(u,v)))
    
    adj = _build_adj(base_edges, n)
    
    # Greedy: repeatedly add the edge that most reduces diameter
    # while maintaining triangle-free
    added = []
    max_iter = 50
    
    for _ in range(max_iter):
        diam = _max_diameter(adj, n)
        if diam <= 4:
            break
        
        # Find all candidate edges (u,v) not in graph, triangle-free safe
        best_edge = None
        best_score = float('inf')
        
        # For efficiency, focus on edges that connect far-apart vertices
        # Generate candidates: all pairs not already connected
        candidates = []
        for u in range(n):
            for v in range(u + 1, n):
                if (u, v) not in norm_edges and (v, u) not in norm_edges:
                    # Check triangle-free
                    if _is_triangle_free(adj, u, v, n):
                        candidates.append((u, v))
        
        # For each candidate, compute resulting diameter (expensive but n=24)
        # Optimize: only evaluate a subset - prioritize long edges
        # Sort candidates by distance in current graph (add long shortcuts first)
        # Compute current distances for sorting
        dist_from_0 = _bfs_distances(adj, 0, n)
        
        # Try all candidates but limit for speed
        # With n=24, max candidates ~276, BFS is fast
        for u, v in candidates:
            # Tentatively add edge
            adj[u].append(v)
            adj[v].append(u)
            norm_edges.add((min(u,v), max(u,v)))
            
            d = _max_diameter(adj, n)
            
            # Remove edge
            adj[u].remove(v)
            adj[v].remove(u)
            norm_edges.discard((min(u,v), max(u,v)))
            
            if d < best_score:
                best_score = d
                best_edge = (u, v)
                if d <= 4:
                    break
        
        if best_edge is None:
            break
        
        u, v = best_edge
        adj[u].append(v)
        adj[v].append(u)
        norm_edges.add((min(u,v), max(u,v)))
        added.append([u, v])
    
    # Return as list of [u, v] with Python ints
    return [[int(u), int(v)] for u, v in added]
# EVOLVE-BLOCK-END
