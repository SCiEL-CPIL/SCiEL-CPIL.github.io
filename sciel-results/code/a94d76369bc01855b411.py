def construct():
    """
    Construct an undirected simple graph on vertices 0..11.
    Two disjoint copies of K_{3,3}.
    """
    edges = []
    
    # Define the two components
    # Component 1: vertices 0-5
    # Bipartition A1: 0, 1, 2
    # Bipartition B1: 3, 4, 5
    for i in range(3):
        for j in range(3):
            u = i
            v = 3 + j
            edges.append([u, v])
            
    # Component 2: vertices 6-11
    # Bipartition A2: 6, 7, 8
    # Bipartition B2: 9, 10, 11
    for i in range(3):
        for j in range(3):
            u = 6 + i
            v = 9 + j
            edges.append([u, v])
            
    # Ensure consistent ordering and type
    final_edges = []
    for e in edges:
        u, v = e
        if u > v:
            u, v = v, u
        final_edges.append([int(u), int(v)])
        
    return final_edges
