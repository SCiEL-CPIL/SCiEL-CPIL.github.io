"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START

def construct():
    # The Fano plane is the proven unique minimum 3-uniform hypergraph
    # with chromatic number 3. It has exactly 7 edges.
    # No 3-uniform hypergraph with fewer than 7 edges is 3-chromatic.
    
    edges = [
        (0, 1, 2),
        (0, 3, 4),
        (0, 5, 6),
        (1, 3, 5),
        (1, 4, 6),
        (2, 3, 6),
        (2, 4, 5)
    ]
    
    # Valid 3-coloring: no edge is monochromatic
    # Vertices 7 and 8 are isolated, so their colors can be anything in 0..2
    colors = (0, 1, 2, 1, 2, 0, 1, 0, 0)
    
    # Return types must be strictly Python int and list/tuple
    return {
        'edges': edges,
        'colors': colors
    }

# EVOLVE-BLOCK-END
