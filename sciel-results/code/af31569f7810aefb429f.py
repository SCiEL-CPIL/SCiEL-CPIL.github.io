def construct():
    """
    Constructs the complete bipartite graph K_{8,8} on 16 vertices.
    
    This graph has 64 edges.
    It is triangle-free, which ensures that every edge is critical for maintaining
    diameter 2. Removing any edge (u, v) will increase the distance between u and v
    to at least 3 (since they share no common neighbors), thus increasing the diameter
    or disconnecting the graph.
    
    By Mantel's Theorem, 64 is the maximum number of edges in a triangle-free graph
    on 16 vertices. Since any triangle would make at least one edge non-critical 
    (providing an alternative path of length 2), the graph must be triangle-free.
    Thus, 64 is the global maximum edge count for this problem.
    """
    import time
    start_time = time.time()
    
    n = 16
    edges = []
    for u in range(8):
        for v in range(8, 16):
            edges.append([int(u), int(v)])
            
    # Deadline guard
    if time.time() - start_time > 85:
        return []
        
    return edges
