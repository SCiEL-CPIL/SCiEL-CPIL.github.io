"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START

def construct():
    # The Fano plane is the unique 3-uniform hypergraph with 7 edges
    # that is 3-chromatic. It is defined on 7 vertices.
    # We embed it in 9 vertices by adding two isolated vertices.
    
    edges = [
        [0, 1, 2],
        [0, 3, 4],
        [0, 5, 6],
        [1, 3, 5],
        [1, 4, 6],
        [2, 3, 6],
        [2, 4, 5]
    ]
    
    # A valid 3-coloring for the Fano plane vertices 0-6
    # v0:0, v1:1, v2:2
    # v3:0, v4:1, v5:2
    # v6:2
    # Vertices 7,8 are isolated, so they can be any color.
    colors = [0, 1, 2, 0, 1, 2, 2, 0, 0]
    
    return {'edges': edges, 'colors': colors}

# EVOLVE-BLOCK-END
