def construct():
    """
    Returns a 3-uniform hypergraph on 9 vertices (0..8) that is 3-chromatic
    with the minimum possible number of edges (7).
    """
    edges = [
        (0, 1, 2),
        (0, 3, 4),
        (0, 5, 6),
        (1, 3, 5),
        (1, 4, 6),
        (2, 3, 6),
        (2, 4, 5)
    ]
    colors = [0, 1, 2, 1, 0, 2, 1, 0, 0]
    return {'edges': edges, 'colors': colors}
