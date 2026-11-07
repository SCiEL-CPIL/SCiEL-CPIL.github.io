import time

def _bfs_maxdist(n, adj, exclude_edge):
    """Return max shortest path distance, or -1 if disconnected."""
    ex_u, ex_v = exclude_edge
    for s in range(n):
        # BFS from s
        dist = [-1] * n
        dist[s] = 0
        queue = [s]
        qi = 0
        while qi < len(queue):
            u = queue[qi]
            qi += 1
            nd = dist[u] + 1
            for v in adj[u]:
                if (u == ex_u and v == ex_v) or (u == ex_v and v == ex_u):
                    continue
                if dist[v] == -1:
                    dist[v] = nd
                    queue.append(v)
        for t in range(n):
            if dist[t] == -1:
                return -1
            if dist[t] > 2:
                return dist[t]
    return 2

def construct():
    start_time = time.time()
    DEADLINE = 80  # seconds, leave safety margin

    n = 16
    half = n // 2
    
    # K_{8,8} gives 64 edges. Let's verify it's edge-critical
    # and check if any modification can do better.
    
    # For K_{8,8}: removing any cross-edge (a,v) means a and v have no common
    # neighbor (a only connects to B, v only connects to A, and the edge (a,v)
    # is the only connection between them). So dist(a,v) = infinity. Critical.
    
    # Can we add intra-edges? Add edge (a1,a2) in A.
    # For it to be critical: a1 and a2 must have no common B-neighbor.
    # In K_{8,8}, N_B(a1) = N_B(a2) = all 8 B-vertices. So we'd need to
    # remove edges to make their B-neighborhoods disjoint.
    # If N_B(a1) has size k and N_B(a2) has size 8-k, we remove (8-k) + k = 8
    # cross-edges to add 1 intra-edge. Net loss of 7.
    
    # What about removing a cross-edge and adding an intra-edge elsewhere?
    # Remove (a,v). Now a has 7 B-neighbors, v has 7 A-neighbors.
    # This doesn't help add intra-edges elsewhere.
    
    # What if we use a non-complete bipartite structure?
    # The key constraint: for every edge (u,v), u and v must have NO common neighbor.
    # This means the graph must be triangle-free!
    # 
    # Proof: If (u,v) is an edge and w is a common neighbor, then after removing
    # (u,v), dist(u,v) <= 2 via w. So (u,v) is NOT critical.
    # Therefore, edge-critical diameter-2 graphs must be triangle-free.
    
    # By Mantel's theorem, a triangle-free graph on 16 vertices has at most
    # floor(16^2/4) = 64 edges. K_{8,8} achieves exactly 64.
    
    # So 64 is the theoretical maximum! K_{8,8} is optimal.
    
    # Let's just return K_{8,8} with a quick verification.
    edges = []
    for u in range(half):
        for v in range(half, n):
            edges.append([int(u), int(v)])
    
    # Quick verification that K_{8,8} is valid
    if time.time() - start_time < DEADLINE:
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        
        # Check diameter is exactly 2
        valid = True
        for s in range(n):
            # Distance 1: neighbors
            d1 = adj[s]
            # Distance 2: neighbors of neighbors (excluding s and d1)
            d2 = set()
            for x in d1:
                d2 |= adj[x]
            d2.discard(s)
            d2 -= d1
            
            if len(d1) + len(d2) != n - 1:
                valid = False
                break
        
        # Check every edge is critical (no triangle)
        for u, v in edges:
            if adj[u] & adj[v]:  # common neighbor => triangle
                valid = False
                break
        
        if not valid:
            # Should never happen for K_{8,8}, but fallback
            pass
    
    return edges
