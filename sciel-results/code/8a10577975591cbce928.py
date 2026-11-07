def construct():
    """
    Constructs a graph on 12 vertices (0..11) with minimum degree 3 and girth >= 4.
    The graph consists of two disjoint copies of K_3,3.
    
    K_3,3 properties:
    - Bipartite (triangle-free, girth = 4).
    - 3-regular (min degree 3).
    - Simple cycles have lengths 4 and 6 only.
    
    Distinct cycle lengths: {4, 6}. Count = 2.
    """
    edges = []
    
    # Component 1: K_{3,3} on vertices 0, 1, 2 (Left) and 3, 4, 5 (Right)
    for u in range(3):
        for v in range(3, 6):
            edges.append([u, v])
    
    # Component 2: K_{3,3} on vertices 6, 7, 8 (Left) and 9, 10, 11 (Right)
    for u in range(6, 9):
        for v in range(9, 12):
            edges.append([u, v])
    
    return edges
