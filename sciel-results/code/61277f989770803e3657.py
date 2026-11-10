import time

def construct():
    """
    Constructs a graph on 16 vertices with diameter exactly 2,
    where every edge is critical (removing any edge increases diameter > 2).
    Returns a list of edges [u, v].
    """
    t0 = time.time()
    deadline = 85.0  # Safety margin before 90s kill
    
    n = 16
    # The optimal graph is the complete bipartite graph K_{8,8}.
    # Vertices 0-7 in set A, 8-15 in set B.
    # Edges connect every vertex in A to every vertex in B.
    # This graph is triangle-free.
    # Diameter is 2.
    # Every edge is critical because removing (u,v) removes the only length-1 path.
    # Since the graph is bipartite and triangle-free, u and v have no common neighbors,
    # so no length-2 path exists. The next shortest path is length 3.
    # Thus diameter increases to 3.
    
    edges = []
    for u in range(8):
        for v in range(8, 16):
            edges.append([u, v])
            
    # Quick verification to ensure correctness
    if time.time() - t0 < deadline:
        if _verify(edges, n):
            return edges
        # Fallback if verification fails (should not happen)
        return [[0, v] for v in range(1, n)]
    
    return edges

def _verify(edges, n):
    """
    Verifies that the graph has diameter exactly 2
    and every edge is critical.
    """
    adj = {v: set() for v in range(n)}
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
        
    def get_diameter():
        max_d = 0
        for s in range(n):
            dist = {s: 0}
            q = [s]
            while q:
                u = q.pop(0)
                for w in adj[u]:
                    if w not in dist:
                        dist[w] = dist[u] + 1
                        q.append(w)
            if len(dist) != n:
                return None # Disconnected
            max_d = max(max_d, max(dist.values()))
        return max_d

    # Check initial diameter is 2
    d = get_diameter()
    if d != 2:
        return False
        
    # Check each edge is critical
    for u, v in edges:
        # Remove edge
        adj[u].remove(v)
        adj[v].remove(u)
        
        d2 = get_diameter()
        
        # Restore edge
        adj[u].add(v)
        adj[v].add(u)
        
        # If d2 is not None and d2 <= 2, the edge was not critical
        if d2 is not None and d2 <= 2:
            return False
            
    return True
