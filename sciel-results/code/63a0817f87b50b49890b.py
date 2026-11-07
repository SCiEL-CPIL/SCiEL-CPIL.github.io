def construct():
    """
    Returns a 3-uniform hypergraph on 9 vertices (0..8) with the minimum number
    of edges such that the hypergraph is 3-chromatic (not 2-colorable, but 3-colorable).
    
    The minimum number of edges is 7, achieved by the Fano plane.
    Vertices 0-6 form the Fano plane. Vertices 7 and 8 are isolated.
    """
    # Fano plane edges
    edges = [
        [0, 1, 2],
        [0, 3, 4],
        [0, 5, 6],
        [1, 3, 5],
        [1, 4, 6],
        [2, 3, 6],
        [2, 4, 5]
    ]
    
    # Valid 3-coloring
    # 0:0, 1:1, 2:2
    # 3:1, 4:2, 5:0
    # 6:1
    # 7:0, 8:0
    colors = [0, 1, 2, 1, 2, 0, 1, 0, 0]
    
    return {
        'edges': edges,
        'colors': colors
    }
