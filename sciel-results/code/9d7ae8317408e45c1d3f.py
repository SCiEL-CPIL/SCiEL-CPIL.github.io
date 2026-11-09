"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START

def construct():
    # The Fano plane is the minimal 3-uniform hypergraph with chromatic number 3.
    # It has 7 vertices and 7 edges.
    # We embed it in 9 vertices by adding two isolated vertices (7 and 8).
    
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
    
    # A valid 3-coloring for the Fano plane
    # Vertices 0..6 are colored 0,1,2,0,1,2,0 respectively.
    # Vertices 7,8 are isolated, color 0.
    colors = [0, 1, 2, 0, 1, 2, 0, 0, 0]
    
    # Ensure types are strict Python int and list
    edges = [list(map(int, e)) for e in edges]
    colors = list(map(int, colors))
    
    return {'edges': edges, 'colors': colors}

# EVOLVE-BLOCK-END
