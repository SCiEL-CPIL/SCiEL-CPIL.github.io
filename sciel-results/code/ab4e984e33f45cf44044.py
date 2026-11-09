"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START

def construct():
    # The Fano plane (PG(2,2)) is the unique minimum 3-uniform hypergraph
    # with chromatic number 3, having exactly 7 edges.
    # No 3-uniform hypergraph with fewer than 7 edges is 3-chromatic.
    # We embed it on vertices 0..6; vertices 7 and 8 are isolated.

    edges = (
        (0, 1, 2),
        (0, 3, 4),
        (0, 5, 6),
        (1, 3, 5),
        (1, 4, 6),
        (2, 3, 6),
        (2, 4, 5),
    )

    # Valid 3-coloring: no edge is monochromatic
    colors = (0, 1, 2, 1, 2, 0, 1, 0, 0)

    return {
        'edges': edges,
        'colors': colors
    }

# EVOLVE-BLOCK-END
