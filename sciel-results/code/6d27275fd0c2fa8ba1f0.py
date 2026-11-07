"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""

# EVOLVE-BLOCK-START

def construct():
    # Fano plane: 3-uniform hypergraph on 7 vertices, 7 edges, chromatic number 3.
    # It is the unique (up to isomorphism) 3-uniform non-2-colorable hypergraph
    # with 7 edges. It is a well-known result that no 3-uniform hypergraph with
    # fewer than 7 edges can be non-2-colorable, so 7 is the minimum.
    #
    # We embed it in 9 vertices (0..8); vertices 7 and 8 are isolated.
    #
    # A valid 3-coloring: assign colors so that every edge has at least two
    # distinct colors.

    edges = [
        [0, 1, 2],
        [0, 3, 4],
        [0, 5, 6],
        [1, 3, 5],
        [1, 4, 6],
        [2, 3, 6],
        [2, 4, 5],
    ]

    # Color assignment for the Fano plane (vertices 0-6), with 7,8 getting color 0.
    # Verify: no edge is monochromatic.
    colors = [0, 1, 2, 1, 2, 0, 1, 0, 0]

    # Build output with exact Python int types and list containers
    edges_out = [[int(v) for v in e] for e in edges]
    colors_out = [int(c) for c in colors]

    return {'edges': edges_out, 'colors': colors_out}

# EVOLVE-BLOCK-END
