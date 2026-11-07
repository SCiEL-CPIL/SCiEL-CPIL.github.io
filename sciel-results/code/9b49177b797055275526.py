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

    # Base: K_{8,8} with partitions A=0..7, B=8..15
    # This gives 64 edges. We know this is valid and has score 64/15 ≈ 4.2667.
    
    # Strategy: Try to find a graph with >64 edges that is edge-critical diameter-2.
    # This is extremely difficult. The theoretical max for edge-critical diameter-2
    # on n vertices is not well-known, but K_{n/2, n/2} is a strong candidate for optimality.
    
    # Let's try a quick local search: start with K_{8,8}, try adding one intra-edge,
    # and see if we can remove fewer cross edges to maintain the property.
    # Actually, as analyzed in the current program, adding one intra-edge (a1, a2) 
    # requires N_B(a1) ∩ N_B(a2) = ∅, which means we must remove at least 
    # (|N_B(a1)| - 1) + (|N_B(a2)| - 1) cross edges if we keep them connected to at least one B.
    # Since |N_B(a)| = 8 for all a in K_{8,8}, we'd remove at least 7+7=14 edges to add 1. Net -13.
    # This is clearly not beneficial.
    
    # What if we start from a different base? 
    # Consider a graph that is not bipartite. 
    # Let's try to construct a graph with one triangle.
    # Vertices 0,1,2 form a triangle: (0,1), (1,2), (2,0).
    # For (0,1) to be critical, dist(0,1) > 2 when removed. So no common neighbor of 0 and 1 other than 2.
    # But 2 is a neighbor of both. So if we remove (0,1), dist(0,1) = 2 via 2. So (0,1) is NOT critical.
    # Therefore, any triangle makes its edges non-critical (since the third vertex is a common neighbor).
    # Thus, the graph must be triangle-free.
    
    # In a triangle-free graph, for any edge (u,v), N(u) ∩ N(v) = ∅.
    # This is a strong constraint.
    # For K_{8,8}, this holds: N_A(u) ∩ N_A(v) = ∅ for u,v in A, and N_B(u) ∩ N_B(v) = ∅ for u,v in B.
    # Wait, in K_{8,8}, for edge (a,b) with a in A, b in B:
    # N(a) = B, N(b) = A. N(a) ∩ N(b) = ∅. Yes.
    # So K_{8,8} is triangle-free and every edge is critical.
    
    # Can we have a triangle-free graph on 16 vertices with more than 64 edges?
    # Mantel's theorem says the max edges in a triangle-free graph on n vertices is floor(n^2/4).
    # For n=16, floor(256/4) = 64.
    # So 64 is the maximum possible for ANY triangle-free graph on 16 vertices.
    # Since any edge-critical diameter-2 graph must be triangle-free (as shown above),
    # 64 is the absolute maximum.
    
    # Therefore, K_{8,8} is optimal. No improvement is possible.
    
    edges = []
    for u in range(half):
        for v in range(half, n):
            edges.append([int(u), int(v)])
    
    return edges
