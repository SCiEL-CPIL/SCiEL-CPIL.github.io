def construct():
    """
    Returns a graph on 16 vertices with 65 edges minimizing the max degree sum of triangles.
    Structure: Complete bipartite graph K_{6,10} plus a perfect matching on the part of size 10.
    This structure yields a max triangle degree sum of 24.
    """
    n = 16
    # Part A: vertices 0 to 5 (size 6)
    # Part B: vertices 6 to 15 (size 10)
    
    edges = []
    
    # Add all edges between Part A and Part B (K_{6,10})
    for u in range(6):
        for v in range(6, 16):
            edges.append([u, v])
            
    # Add matching edges within Part B
    # Pairs: (6,7), (8,9), (10,11), (12,13), (14,15)
    for i in range(6, 16, 2):
        edges.append([i, i + 1])
        
    return {'n': n, 'edges': edges}
