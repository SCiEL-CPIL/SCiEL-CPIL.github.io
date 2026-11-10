def construct():
    """
    Constructs a graph on 12 vertices (0-11) with minimum degree 3,
    triangle-free, and minimizing the number of distinct cycle lengths.
    
    The graph consists of two disjoint copies of K_3,3.
    K_3,3 is bipartite with parts of size 3.
    It has girth 4.
    The cycles in K_3,3 have lengths 4 and 6.
    Thus, the distinct cycle lengths are {4, 6}, count = 2.
    
    This is optimal because a connected graph with min degree 3 and
    triangle-free must have at least 2 distinct cycle lengths.
    """
    n = 12
    edges = []
    
    # First K_3,3 on vertices 0-5
    # Left part: 0, 1, 2
    # Right part: 3, 4, 5
    left_1 = [0, 1, 2]
    right_1 = [3, 4, 5]
    for u in left_1:
        for v in right_1:
            edges.append([u, v])
            
    # Second K_3,3 on vertices 6-11
    # Left part: 6, 7, 8
    # Right part: 9, 10, 11
    left_2 = [6, 7, 8]
    right_2 = [9, 10, 11]
    for u in left_2:
        for v in right_2:
            edges.append([u, v])
            
    return edges
