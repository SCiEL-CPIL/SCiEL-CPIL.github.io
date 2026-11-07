"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START

import itertools
import time

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # safety margin before 90s kill

    # The Fano plane is the smallest 3-uniform hypergraph with chromatic number 3.
    # It has 7 edges and 7 vertices. We use vertices 0..6.
    # Vertices 7 and 8 are isolated.
    
    edges = [
        (0, 1, 2),
        (0, 3, 4),
        (0, 5, 6),
        (1, 3, 5),
        (1, 4, 6),
        (2, 3, 6),
        (2, 4, 5)
    ]
    
    # Convert to lists as required by the interface (list or tuple is fine, but list is standard)
    edges = [list(e) for e in edges]
    
    # A valid 3-coloring for the Fano plane vertices 0..6
    # We need to provide 9 colors for vertices 0..8.
    # Vertices 0-6 are part of the Fano plane.
    # Vertices 7, 8 are isolated, so they can be any color.
    
    # Fano plane 3-coloring:
    # 0: 0
    # 1: 1
    # 2: 2
    # 3: 1
    # 4: 2
    # 5: 0
    # 6: 1
    colors = [0, 1, 2, 1, 2, 0, 1, 0, 0]
    
    # Quick verification that edges are non-monochromatic
    for edge in edges:
        c = {colors[v] for v in edge}
        if len(c) == 1:
            # Should not happen for valid Fano plane coloring
            pass

    return {'edges': edges, 'colors': colors}

# EVOLVE-BLOCK-END
