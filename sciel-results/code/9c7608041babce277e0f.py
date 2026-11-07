def construct():
    """
    Constructs a graph on 12 vertices (0-11) as a complete bipartite graph K_{6,6}.
    
    Properties:
    - Vertices 0-5 form one partition.
    - Vertices 6-11 form the other partition.
    - All possible edges between partitions exist.
    
    Each vertex has degree 6.
    The graph is triangle-free (bipartite), so girth is 4.
    The simple cycles in K_{6,6} have lengths 4, 6, 8, 10, 12.
    Thus, the set of distinct simple cycle lengths is {4, 6, 8, 10, 12}, count = 5.
    """
    edges = []
    
    # K_{6,6} component: {0, 1, 2, 3, 4, 5} x {6, 7, 8, 9, 10, 11}
    for u in range(6):
        for v in range(6, 12):
            edges.append([u, v])
            
    return edges
