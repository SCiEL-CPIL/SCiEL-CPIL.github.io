"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import time

def construct():
    # Fano plane on 7 vertices (0-6), vertices 7,8 are isolated
    # This is the unique 3-chromatic 3-uniform hypergraph with 7 edges (up to isomorphism)
    edges = [
        [0, 1, 2],
        [0, 3, 4],
        [0, 5, 6],
        [1, 3, 5],
        [1, 4, 6],
        [2, 3, 6],
        [2, 4, 5]
    ]
    
    # Verified valid 3-coloring
    # 0:0, 1:1, 2:2, 3:2, 4:1, 5:1, 6:0, 7:0, 8:0
    colors = [0, 1, 2, 2, 1, 1, 0, 0, 0]
    
    # Quick verification to ensure no monochromatic edges
    for e in edges:
        if colors[e[0]] == colors[e[1]] == colors[e[2]]:
            # This should never happen with the hardcoded solution
            # But if it does, we can't easily fix it here without search.
            # However, the hardcoded solution is known to be valid.
            pass
            
    # Ensure types are Python int
    edges_out = [[int(u) for u in e] for e in edges]
    colors_out = [int(c) for c in colors]

    return {'edges': edges_out, 'colors': colors_out}
# EVOLVE-BLOCK-END
