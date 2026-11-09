def construct():
    # Fano plane (7 vertices, 7 edges) embedded in 0..6, vertices 7,8 isolated.
    # This is provably minimal: any 3-uniform 3-chromatic hypergraph needs >= 7 edges,
    # and the Fano plane achieves it.
    # We use tuples for edges (distinct, nonmonochromatic) and ensure types are pure Python int.
    edges = (
        (0, 1, 2),
        (0, 3, 4),
        (0, 5, 6),
        (1, 3, 5),
        (1, 4, 6),
        (2, 3, 6),
        (2, 4, 5),
    )
    colors = (0, 1, 2, 0, 1, 1, 0, 0, 0)
    return {'edges': list(edges), 'colors': list(colors)}
