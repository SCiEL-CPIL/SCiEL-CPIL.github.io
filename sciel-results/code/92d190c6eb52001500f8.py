"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START

def construct():
    """
    Constructs a 3-uniform hypergraph on 9 vertices (0..8) that is 3-chromatic
    with the minimum possible number of edges (7).
    
    The core structure is the Fano plane (PG(2,2)) embedded on vertices 0-6.
    Vertices 7 and 8 are isolated.
    
    The Fano plane is the smallest 3-uniform hypergraph that is not 2-colorable.
    It has exactly 7 edges. Since any 3-chromatic 3-uniform hypergraph must
    contain a non-2-colorable sub-hypergraph, and the Fano plane is the minimal
    such structure, 7 is the theoretical minimum edge count.
    """
    
    # The 7 edges of the Fano plane on vertices 0..6
    # Derived from {i, j, i^j} for i, j in 1..7
    edges = [
        [0, 1, 2],
        [0, 3, 4],
        [0, 5, 6],
        [1, 3, 5],
        [1, 4, 6],
        [2, 3, 6],
        [2, 4, 5]
    ]
    
    # A valid 3-coloring where no edge is monochromatic
    # Vertices 0..6 form the Fano plane. 7 and 8 are isolated.
    colors = [0, 1, 2, 0, 1, 2, 0, 0, 0]
    
    # Ensure types are strictly Python int and list
    edges_out = [[int(v) for v in e] for e in edges]
    colors_out = [int(c) for c in colors]
    
    return {
        'edges': edges_out,
        'colors': colors_out
    }

# EVOLVE-BLOCK-END
