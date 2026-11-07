"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from collections import deque

P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}

def construct():
    n = P['n']
    base_edges = set()
    for e in P['edges']:
        base_edges.add((e[0], e[1]))
        base_edges.add((e[1], e[0]))
    
    adj = [set() for _ in range(n)]
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)
    
    def bfs_diameter(adj):
        max_d = 0
        for src in range(n):
            dist = [-1] * n
            dist[src] = 0
            q = deque([src])
            while q:
                u = q.popleft()
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        max_d = max(max_d, dist[v])
                        q.append(v)
        return max_d
    
    # Only add even-odd edges to maintain bipartite (triangle-free)
    candidates = []
    for u in range(n):
        for v in range(u + 1, n):
            if (u % 2) != (v % 2) and (u, v) not in base_edges:
                candidates.append((u, v))
    
    added = []
    current_diam = bfs_diameter(adj)
    deadline = time.time() + 55
    
    while current_diam > 4 and time.time() < deadline:
        best_edge = None
        best_diam = current_diam
        for (u, v) in candidates:
            if (u, v) in base_edges or (v, u) in base_edges:
                continue
            adj[u].add(v)
            adj[v].add(u)
            d = bfs_diameter(adj)
            adj[u].discard(v)
            adj[v].discard(u)
            if d < best_diam:
                best_diam = d
                best_edge = (u, v)
        
        if best_edge is None:
            break
        
        u, v = best_edge
        adj[u].add(v)
        adj[v].add(u)
        added.append([u, v])
        current_diam = best_diam
    
    return added
# EVOLVE-BLOCK-END

