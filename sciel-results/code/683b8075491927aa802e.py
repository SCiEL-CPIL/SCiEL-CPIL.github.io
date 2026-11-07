"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START

def construct():
    # The Fano plane is the smallest 3-chromatic 3-uniform hypergraph.
    # It has 7 vertices and 7 edges.
    # We define it on vertices 0..6. Vertices 7 and 8 are isolated.
    # This satisfies the requirement of a hypergraph on 9 vertices (0..8)
    # because the vertex set is fixed, but edges can only involve a subset.
    
    edges = [
        (0, 1, 2),
        (0, 3, 4),
        (0, 5, 6),
        (1, 3, 5),
        (1, 4, 6),
        (2, 3, 6),
        (2, 4, 5)
    ]
    
    # A valid 3-coloring for the Fano plane where no edge is monochromatic.
    # v0=0, v1=1, v2=2, v3=1, v4=2, v5=0, v6=1
    # v7=0, v8=0 (isolated, color doesn't affect edge constraints)
    colors = [0, 1, 2, 1, 2, 0, 1, 0, 0]
    
    # Convert edges to lists as required by the interface check 
    # (though tuple is also allowed, list is safer if strict list check is implied 
    # but task says "list or tuple". Let's use list to be safe).
    edges_list = [list(e) for e in edges]
    
    return {'edges': edges_list, 'colors': colors}

# EVOLVE-BLOCK-END
