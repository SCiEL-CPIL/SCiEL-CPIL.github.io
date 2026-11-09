"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    # Fano plane on vertices 0..6
    edges = [
        (0, 1, 2),
        (0, 3, 4),
        (0, 5, 6),
        (1, 3, 5),
        (1, 4, 6),
        (2, 3, 6),
        (2, 4, 5)
    ]
    # 3-coloring for Fano plane
    colors = [0, 1, 2, 1, 2, 0, 1, 0, 0]
    
    # Convert to lists of ints as required
    edges_list = [list(e) for e in edges]
    colors_list = [int(c) for c in colors]
    
    return {'edges': edges_list, 'colors': colors_list}
# EVOLVE-BLOCK-END
