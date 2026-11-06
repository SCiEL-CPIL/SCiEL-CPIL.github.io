def _groetzsch_edge_mask():
    """
    Return the edge mask for the Grötzsch graph (Mycielski of C5) on 11 vertices.
    Vertices 0-10 are used. 11-15 are isolated initially.
    This graph is triangle-free and 4-chromatic.
    """
    n = 16
    mask = 0
    
    # 1. Cycle C5: 0-1-2-3-4-0
    cycle_edges = [(0,1), (1,2), (2,3), (3,4), (4,0)]
    for u, v in cycle_edges:
        mask |= 1 << (u * n + v)
        
    # 2. Mycielski edges: for each edge (u,v) in C5, add (u, v+5) and (v, u+5)
    for u, v in cycle_edges:
        mask |= 1 << (u * n + (v + 5))
        mask |= 1 << (v * n + (u + 5))
        
    # 3. Apex vertex 10 connected to all copies 5-9
    for u in range(5, 10):
        mask |= 1 << (10 * n + u)
        
    return mask
