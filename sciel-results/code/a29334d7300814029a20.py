def construct():
    """
    Constructs a simple undirected graph on 16 vertices (0..15) with:
    1. Diameter exactly 2.
    2. Every edge is critical: deleting any single edge increases the diameter 
       above 2 or disconnects the graph.
    3. Maximized edge count.
    
    Strategy:
    - To satisfy edge criticality with diameter 2, the graph must be triangle-free.
      (If a triangle exists, removing an edge from it leaves the endpoints connected 
       via the third vertex with distance 2, so the diameter does not increase, 
       and the graph remains connected, violating the 'critical' condition).
    - By Mantel's Theorem, the maximum number of edges in a triangle-free graph 
      on n vertices is floor(n^2 / 4).
    - For n=16, the maximum is 64, achieved uniquely by the complete bipartite 
      graph K_{8,8}.
    - In K_{8,8}, any edge (u, v) connects the two partitions. Removing it forces 
      the distance between u and v to become 3 (via any other node in the opposite 
      partition and back), thus increasing the diameter to 3.
    """
    n = 16
    half = n // 2
    edges = []
    
    # Construct K_{8,8}
    # Left partition: 0..7
    # Right partition: 8..15
    for u in range(half):
        for v in range(half, n):
            # Ensure types are strictly Python int
            edges.append([int(u), int(v)])
            
    return edges
