"""
Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK.
"""

# EVOLVE-BLOCK-START

def construct():
    # The minimum number of edges in a 3-uniform hypergraph that is 3-chromatic
    # is 7, achieved by the Fano plane. Any 3-uniform hypergraph with <= 6 edges
    # is 2-colorable (can be 2-colored without monochromatic edges).
    #
    # The Fano plane has 7 vertices and 7 edges. We embed it in 9 vertices (0..8)
    # by using vertices 0..6 for the Fano plane and leaving vertices 7 and 8 isolated.
    #
    # The Fano plane edges (standard representation):
    # Each edge is a triple of vertices.
    
    edges = [
        [0, 1, 2],
        [0, 3, 4],
        [0, 5, 6],
        [1, 3, 5],
        [1, 4, 6],
        [2, 3, 6],
        [2, 4, 5]
    ]
    
    # Verify that this is indeed the Fano plane:
    # - 7 edges, each of size 3
    # - Every pair of vertices appears in exactly one edge (21 pairs, 7 edges × 3 pairs per edge = 21) ✓
    # - The hypergraph is not 2-colorable (it's 3-chromatic)
    #
    # A valid 3-coloring:
    # We need to assign colors from {0, 1, 2} to vertices 0..8 such that
    # no edge is monochromatic.
    #
    # Let's verify the coloring [0, 1, 2, 1, 2, 0, 1, 0, 0]:
    # Edge [0,1,2]: colors 0,1,2 - not monochromatic ✓
    # Edge [0,3,4]: colors 0,1,2 - not monochromatic ✓
    # Edge [0,5,6]: colors 0,0,1 - not monochromatic ✓
    # Edge [1,3,5]: colors 1,1,0 - not monochromatic ✓
    # Edge [1,4,6]: colors 1,2,1 - not monochromatic ✓
    # Edge [2,3,6]: colors 2,1,1 - not monochromatic ✓
    # Edge [2,4,5]: colors 2,2,0 - not monochromatic ✓
    #
    # All edges are non-monochromatic. ✓
    #
    # The Fano plane is known to be 3-chromatic: it cannot be 2-colored without
    # having a monochromatic edge. This is a well-known result in hypergraph theory.
    
    colors = [0, 1, 2, 1, 2, 0, 1, 0, 0]
    
    # Ensure all types are Python native types
    edges_out = [list(int(x) for x in e) for e in edges]
    colors_out = [int(c) for c in colors]
    
    return {'edges': edges_out, 'colors': colors_out}

# EVOLVE-BLOCK-END
