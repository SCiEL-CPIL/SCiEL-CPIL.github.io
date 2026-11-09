"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START

def construct():
    """
    Constructs the minimal 3-uniform 3-chromatic hypergraph on 9 vertices.
    The Fano plane on 7 vertices is the unique minimal 3-uniform 3-chromatic hypergraph.
    We use vertices 0-6 for the Fano plane and leave 7-8 isolated.
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
    
    return {'edges': edges, 'colors': colors}

# EVOLVE-BLOCK-END
