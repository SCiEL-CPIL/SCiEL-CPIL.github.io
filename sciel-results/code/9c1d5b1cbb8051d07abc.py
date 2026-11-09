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
    t0 = time.time()
    n = 16
    half = n // 2
    
    # Structural Change: Use set-based construction for clarity and robustness
    # Partition vertices into two sets A and B
    A = set(range(half))
    B = set(range(half, n))
    
    edge_set = set()
    for u in A:
        for v in B:
            # Ensure u < v for canonical representation, though order doesn't matter for undirected
            if u < v:
                edge_set.add((u, v))
            else:
                edge_set.add((v, u))
                
    # Convert to list of lists of ints
    edge_list = [[int(u), int(v)] for u, v in edge_set]
    
    # Build adjacency list for verification
    adj = [[] for _ in range(n)]
    for u, v in edge_list:
        adj[u].append(v)
        adj[v].append(u)
        
    # Verify diameter is exactly 2
    if _bfs_diameter(adj) != 2:
        return []
        
    # Deadline guard
    if time.time() - t0 > 55:
        return edge_list

    return edge_list
