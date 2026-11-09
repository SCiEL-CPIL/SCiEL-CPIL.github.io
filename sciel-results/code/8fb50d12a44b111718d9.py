import time
from collections import deque

def _calc_diameter(adj):
    n = len(adj)
    max_dist = 0
    for s in range(n):
        dist = [-1] * n
        dist[s] = 0
        q = deque([s])
        while q:
            u = q.popleft()
            for v in adj[u]:
                if dist[v] < 0:
                    dist[v] = dist[u] + 1
                    q.append(v)
        for d in dist:
            if d < 0:
                return float('inf')
            if d > max_dist:
                max_dist = d
    return max_dist

def construct():
    start_time = time.time()
    n = 16
    edges = []
    
    # Construct K_{8,8}
    # Part 1: 0..7, Part 2: 8..15
    for i in range(8):
        for j in range(8, 16):
            edges.append([i, j])
            
    # Build adjacency list for verification
    adj = [[] for _ in range(n)]
    for e in edges:
        u, v = e
        adj[u].append(v)
        adj[v].append(u)
        
    # Verify diameter is 2
    d = _calc_diameter(adj)
    if d != 2:
        return []
        
    # Verify edge essentiality (triangle-free implies essential for diam 2)
    # Check if any edge lies in a triangle.
    # For each edge (u,v), check if N(u) and N(v) intersect.
    for u in range(n):
        for v in adj[u]:
            if u > v: continue # check each edge once
            # Check common neighbors
            common = set(adj[u]).intersection(adj[v])
            if len(common) > 0:
                # Edge is not essential
                return []
                
    if time.time() - start_time > 55:
        return edges
        
    return edges
