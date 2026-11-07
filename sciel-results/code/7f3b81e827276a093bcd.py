"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from collections import deque

P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}

def construct():
    n = P['n']
    base_edges = set()
    for e in P['edges']:
        base_edges.add(frozenset(e))
    
    # Build adjacency for triangle check
    adj = [set() for _ in range(n)]
    for e in P['edges']:
        adj[e[0]].add(e[1])
        adj[e[1]].add(e[0])
    
    def is_triangle_free(adj, u, v):
        # Check if adding edge (u,v) creates a triangle
        for w in adj[u]:
            if w in adj[v]:
                return False
        return True
    
    def compute_diameter(adj):
        max_d = 0
        for s in range(n):
            dist = [-1] * n
            dist[s] = 0
            q = deque([s])
            while q:
                u = q.popleft()
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        q.append(v)
            for d in dist:
                if d > max_d:
                    max_d = d
        return max_d
    
    # Greedy: add edges to reduce diameter
    added = []
    deadline = time.time() + 50
    
    # Candidate edges: all non-base edges
    candidates = []
    for u in range(n):
        for v in range(u+1, n):
            if frozenset((u,v)) not in base_edges:
                candidates.append((u,v))
    
    # Sort candidates by some heuristic: prefer edges that connect far-apart vertices
    candidates.sort(key=lambda e: e[1]-e[0], reverse=True)
    
    while compute_diameter(adj) > 4 and time.time() < deadline:
        best_edge = None
        best_diam = compute_diameter(adj)
        
        for u, v in candidates:
            if (u,v) in [(a,b) for a,b in added]:
                continue
            if not is_triangle_free(adj, u, v):
                continue
            
            # Try adding this edge
            adj[u].add(v)
            adj[v].add(u)
            d = compute_diameter(adj)
            
            if d < best_diam:
                best_diam = d
                best_edge = (u,v)
            
            # Undo
            adj[u].remove(v)
            adj[v].remove(u)
        
        if best_edge is None:
            break
        
        u, v = best_edge
        adj[u].add(v)
        adj[v].add(u)
        added.append([u, v])
    
    # Verify
    if compute_diameter(adj) > 4:
        # Fallback: use the bipartite construction
        added = []
        adj = [set() for _ in range(n)]
        for e in P['edges']:
            adj[e[0]].add(e[1])
            adj[e[1]].add(e[0])
        for u in range(n):
            for v in range(u+1, n):
                if (u-v) % 2 == 1 and frozenset((u,v)) not in base_edges:
                    adj[u].add(v)
                    adj[v].add(u)
                    added.append([u, v])
    
    return added
# EVOLVE-BLOCK-END
