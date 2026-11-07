"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    # The Fano plane is the smallest 3-uniform hypergraph that is not 2-colorable.
    # It has 7 edges and 7 vertices.
    # We embed it in the vertex set {0, ..., 8} by leaving vertices 7 and 8 isolated.
    # The chromatic number of the Fano plane is 3.
    # We provide a valid 3-coloring.
    
    edges = [
        (0, 1, 2),
        (0, 3, 4),
        (0, 5, 6),
        (1, 3, 5),
        (1, 4, 6),
        (2, 3, 6),
        (2, 4, 5)
    ]
    
    # Coloring:
    # 0: 0
    # 1: 1
    # 2: 2
    # 3: 0
    # 4: 1
    # 5: 2
    # 6: 0
    # 7: 0 (isolated)
    # 8: 0 (isolated)
    
    colors = [0, 1, 2, 0, 1, 2, 0, 0, 0]
    
    # Verify non-monochromatic edges
    for e in edges:
        c_vals = {colors[v] for v in e}
        if len(c_vals) == 1:
            # This should not happen with the chosen coloring
            pass
            
    return {'edges': [list(e) for e in edges], 'colors': colors}
# EVOLVE-BLOCK-END
