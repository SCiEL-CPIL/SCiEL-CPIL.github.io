"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START

def construct():
    """
    Returns a 3-uniform hypergraph on vertices 0..8 that is exactly 3-chromatic.
    The Fano plane is the unique minimal 3-uniform hypergraph with chromatic number 3.
    It has 7 vertices and 7 edges. Vertices 7 and 8 are isolated.
    """
    # Fano plane edges on vertices 0..6
    edges = [
        [0, 1, 2],
        [0, 3, 4],
        [0, 5, 6],
        [1, 3, 5],
        [1, 4, 6],
        [2, 3, 6],
        [2, 4, 5]
    ]
    
    # A valid 3-coloring for the Fano plane vertices 0..6:
    # The Fano plane is 3-chromatic. One such coloring is:
    # 0: 0
    # 1: 1
    # 2: 2
    # 3: 1
    # 4: 2
    # 5: 0
    # 6: 1
    # Let's verify non-monochromaticity for each edge:
    # [0, 1, 2] -> [0, 1, 2] (OK)
    # [0, 3, 4] -> [0, 1, 2] (OK)
    # [0, 5, 6] -> [0, 0, 1] (OK)
    # [1, 3, 5] -> [1, 1, 0] (OK)
    # [1, 4, 6] -> [1, 2, 1] (OK)
    # [2, 3, 6] -> [2, 1, 1] (OK)
    # [2, 4, 5] -> [2, 2, 0] (OK)
    
    colors = [0, 1, 2, 1, 2, 0, 1, 0, 0]
    
    return {'edges': edges, 'colors': colors}

# EVOLVE-BLOCK-END
