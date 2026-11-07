"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START

def construct():
    # Fano plane edges on vertices 0..6
    # These 7 edges form a 3-uniform hypergraph that is not 2-colorable.
    # Thus, it is 3-chromatic.
    # Vertices 7 and 8 are isolated, so they don't affect the chromatic number.
    edges = [
        [0, 1, 2],
        [0, 3, 4],
        [0, 5, 6],
        [1, 3, 5],
        [1, 4, 6],
        [2, 3, 6],
        [2, 4, 5]
    ]
    
    # A valid 3-coloring for the Fano plane:
    # 0:0, 1:1, 2:2, 3:0, 4:1, 5:2, 6:0
    # Check edges:
    # [0,1,2] -> 0,1,2 (non-mono)
    # [0,3,4] -> 0,0,1 (non-mono)
    # [0,5,6] -> 0,2,0 (non-mono)
    # [1,3,5] -> 1,0,2 (non-mono)
    # [1,4,6] -> 1,1,0 (non-mono)
    # [2,3,6] -> 2,0,0 (non-mono)
    # [2,4,5] -> 2,1,2 (non-mono)
    colors = [0, 1, 2, 0, 1, 2, 0, 0, 0]
    
    return {'edges': edges, 'colors': colors}

# EVOLVE-BLOCK-END
