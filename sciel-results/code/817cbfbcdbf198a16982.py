"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START

def construct():
    # The Fano plane is the unique minimal 3-uniform 3-chromatic hypergraph.
    # It has exactly 7 edges. This is the theoretical minimum.
    # Score = 10 / 7 ≈ 1.42857.
    
    # Vertices 0..6 form the Fano plane. 7 and 8 are isolated.
    edges = (
        (0, 1, 2),
        (0, 3, 4),
        (0, 5, 6),
        (1, 3, 5),
        (1, 4, 6),
        (2, 3, 6),
        (2, 4, 5)
    )
    
    # Valid 3-coloring: 
    # 0:0, 1:1, 2:2
    # 3:1, 4:2, 5:0
    # 6:1
    # 7:0, 8:0
    colors = (0, 1, 2, 1, 2, 0, 1, 0, 0)
    
    # Convert to lists of ints as required by the evaluator
    edges = [list(e) for e in edges]
    colors = list(colors)
    
    return {'edges': edges, 'colors': colors}

# EVOLVE-BLOCK-END
