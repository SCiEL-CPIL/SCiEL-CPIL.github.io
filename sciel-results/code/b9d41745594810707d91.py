"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START

import time

def construct():
    # The Fano plane is the smallest known 3-uniform hypergraph with chromatic number 3.
    # It consists of 7 edges on 7 vertices.
    # We map these to vertices 0..6. Vertices 7 and 8 are isolated.
    
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
    # Vertices: 0, 1, 2, 3, 4, 5, 6
    # Colors:   0, 1, 2, 1, 2, 0, 1
    # Check edge (0,1,2): 0,1,2 (ok)
    # Check edge (0,3,4): 0,1,2 (ok)
    # Check edge (0,5,6): 0,0,1 (ok)
    # Check edge (1,3,5): 1,1,0 (ok)
    # Check edge (1,4,6): 1,2,1 (ok)
    # Check edge (2,3,6): 2,1,1 (ok)
    # Check edge (2,4,5): 2,2,0 (ok)
    
    colors = [0, 1, 2, 1, 2, 0, 1, 0, 0]

    return {'edges': edges, 'colors': colors}

# EVOLVE-BLOCK-END
