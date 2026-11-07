"""
Few hyperedges forcing three colors.

We return the Fano plane (7 edges on 7 of the 9 vertices), which is the
edge-minimal 3-uniform hypergraph that is 3-chromatic.  Vertices 7 and 8 are
isolated.  A valid 3-coloring is provided; every edge is non-monochromatic.
"""

def construct():
    # Fano plane edges (triples) on vertices 0..6
    edges = [
        (0, 1, 2),
        (0, 3, 4),
        (0, 5, 6),
        (1, 3, 5),
        (1, 4, 6),
        (2, 3, 6),
        (2, 4, 5),
    ]

    # A 3-coloring: each edge must contain at least two distinct colors.
    colors = [0, 1, 2, 1, 2, 0, 1, 0, 0]

    return {"edges": list(edges), "colors": list(colors)}
