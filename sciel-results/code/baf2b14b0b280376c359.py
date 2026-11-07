def construct():
    """
    Constructs a maximal edge-critical diameter-2 graph on 16 vertices.
    
    The optimal graph is the complete bipartite graph K_{8,8}.
    - Vertices are partitioned into two sets A and B of size 8 each.
    - All edges connect a vertex in A to a vertex in B.
    - Total edges: 8 * 8 = 64.
    - Diameter is 2.
    - Edge-critical: Removing any edge (u, v) where u in A and v in B 
      leaves u and v with no common neighbors, so the pair becomes disconnected.
    """
    import time
    start_time = time.time()
    deadline = start_time + 85.0  # safety margin before 90s limit
    
    n = 16
    half = n // 2
    
    # Partition vertices
    set_a = list(range(half))
    set_b = list(range(half, n))
    
    edges = []
    
    # Generate all edges between A and B
    for u in set_a:
        for v in set_b:
            edges.append([int(u), int(v)])
    
    # Deadline guard: if somehow we exceeded our budget, return what we have
    if time.time() > deadline:
        return edges
    
    return edges
