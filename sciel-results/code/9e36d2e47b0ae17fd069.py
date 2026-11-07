def construct():
    # Fano plane hypergraph on vertices 0..6; vertices 7,8 are isolated.
    # The Fano plane is a 3-uniform hypergraph with 7 edges that is
    # exactly 3-chromatic: it is not 2-colorable, but a 3-coloring exists.
    # This is the minimum number of edges for a 3-chromatic 3-uniform hypergraph.
    edges = [
        (0, 1, 3),
        (0, 2, 4),
        (0, 5, 6),
        (1, 2, 5),
        (1, 4, 6),
        (2, 3, 6),
        (3, 4, 5)
    ]
    # A valid 3-coloring: each edge contains vertices of at least two colors.
    colors = [0, 1, 2, 0, 1, 2, 0, 0, 0]
    # Ensure all values are Python int (not numpy or bool)
    edges = [tuple(int(v) for v in e) for e in edges]
    colors = [int(c) for c in colors]
    return {'edges': edges, 'colors': colors}
