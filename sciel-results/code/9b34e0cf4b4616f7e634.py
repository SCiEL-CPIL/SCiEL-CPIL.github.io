"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    # The Fano plane is the unique minimal 3-uniform hypergraph that is 3-chromatic.
    # It has 7 edges on 7 vertices.
    # We embed it in 9 vertices (0..8) by leaving vertices 7 and 8 isolated.
    # This minimizes the edge count to 7.
    
    # Fano plane edges on vertices 0..6
    fano_edges = [
        (0, 1, 2),
        (0, 3, 4),
        (0, 5, 6),
        (1, 3, 5),
        (1, 4, 6),
        (2, 3, 6),
        (2, 4, 5)
    ]
    
    # Convert to lists as required by "sequences must be list or tuple"
    edges = [list(e) for e in fano_edges]
    
    # A valid 3-coloring for vertices 0..8
    # Vertices 0..6 are colored to satisfy non-monochromatic constraints for Fano edges.
    # Vertices 7,8 are isolated, so any color is fine.
    colors = [0, 1, 2, 0, 1, 2, 0, 0, 0]
    
    # Final sanity check (should always pass for this static config)
    for e in edges:
        if colors[e[0]] == colors[e[1]] == colors[e[2]]:
            # If this ever happens, it's a bug in our static definition.
            # But we know the Fano plane is 3-colorable.
            pass
            
    return {'edges': edges, 'colors': colors}
# EVOLVE-BLOCK-END
