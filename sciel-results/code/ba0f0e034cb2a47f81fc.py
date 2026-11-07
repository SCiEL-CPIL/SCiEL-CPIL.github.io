"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START

import time

def construct():
    # Deadline guard as per directive
    start_time = time.time()
    deadline = start_time + 50.0  # 50s margin out of 60s
    
    # The Fano plane is the smallest 3-uniform hypergraph with chromatic number 3.
    # It has 7 edges. We embed it in the 9-vertex space {0..8} using vertices 0..6.
    # Vertices 7 and 8 are isolated and can have any color.
    
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
    # 0:0, 1:1, 2:2, 3:1, 4:2, 5:0, 6:1
    colors = [0, 1, 2, 1, 2, 0, 1, 0, 0]
    
    # Ensure types are strictly Python int and list
    edges_out = [tuple(int(v) for v in e) for e in edges]
    colors_out = [int(c) for c in colors]
    
    return {'edges': edges_out, 'colors': colors_out}

# EVOLVE-BLOCK-END
