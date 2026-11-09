"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START

def construct():
    # Fano plane: projective plane of order 2, 7 points, 7 lines (edges).
    # Each line has 3 points; each pair of points is on exactly one line.
    # This is the unique 3-uniform 3-chromatic hypergraph with 7 edges,
    # which is the proven minimum for a 3-chromatic triple system.
    #
    # Representation: points are elements of GF(2)^3 minus {0}, i.e.,
    # the 7 nonzero vectors. Lines are 1-dimensional subspaces (each
    # contains 3 nonzero points: a, a, a+... wait, in GF(2) each line
    # through origin has 2 nonzero points plus... Let me use the standard
    # Fano plane construction.
    #
    # Standard Fano plane edges (0-indexed):
    edges = [
        (0, 1, 2),
        (0, 3, 4),
        (0, 5, 6),
        (1, 3, 5),
        (1, 4, 6),
        (2, 3, 6),
        (2, 4, 5)
    ]
    
    # 3-coloring: assign colors so no edge is monochromatic.
    # Using the fact that Fano plane is 3-colorable:
    # Color class 0: {0, 5} -> color 0
    # Color class 1: {1, 3, 6} -> color 1  
    # Color class 2: {2, 4} -> color 2
    # Check: 
    #   (0,1,2): 0,1,2 - ok
    #   (0,3,4): 0,1,2 - ok
    #   (0,5,6): 0,0,1 - ok
    #   (1,3,5): 1,1,0 - ok
    #   (1,4,6): 1,2,1 - ok
    #   (2,3,6): 2,1,1 - ok
    #   (2,4,5): 2,2,0 - ok
    colors = [0, 1, 2, 1, 2, 0, 1, 0, 0]
    
    # Verify: no edge is monochromatic in this coloring
    for e in edges:
        assert len(set(colors[v] for v in e)) >= 2, "Monochromatic edge found"
    
    # Verify: no 2-coloring exists (Fano plane is 3-chromatic)
    # For a 3-uniform hypergraph to be 2-colorable, there must exist
    # an assignment of 0/1 to vertices such that every edge has both colors.
    # The Fano plane is known to be 3-chromatic (not 2-colorable).
    # Quick verification: try all 2^7 = 128 colorings of vertices 0-6.
    # Vertices 7,8 are isolated and don't affect colorability.
    two_colorable = False
    for mask in range(128):
        all_edges_bipartite = True
        for e in edges:
            bits = [(mask >> v) & 1 for v in e]
            if bits[0] == bits[1] == bits[2]:
                all_edges_bipartite = False
                break
        if all_edges_bipartite:
            two_colorable = True
            break
    assert not two_colorable, "Fano plane should not be 2-colorable"
    
    # Return with correct types
    return {
        'edges': [list(e) for e in edges],
        'colors': list(colors)
    }

# EVOLVE-BLOCK-END
