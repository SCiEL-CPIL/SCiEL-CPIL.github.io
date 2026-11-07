"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    # The Fano plane on 7 vertices is the minimal 3-chromatic 3-uniform hypergraph.
    # We extend it to 9 vertices by adding two isolated vertices (7, 8).
    # This preserves the chromatic number of 3 and minimizes the edge count to 7.
    
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
    
    # A valid 3-coloring for the Fano plane
    # c0=0, c1=1, c2=2, c3=1, c4=2, c5=0, c6=1
    # Check:
    # (0,1,2) -> 0,1,2 OK
    # (0,3,4) -> 0,1,2 OK
    # (0,5,6) -> 0,0,1 OK
    # (1,3,5) -> 1,1,0 OK
    # (1,4,6) -> 1,2,1 OK
    # (2,3,6) -> 2,1,1 OK
    # (2,4,5) -> 2,2,0 OK
    
    colors = [0, 1, 2, 1, 2, 0, 1, 0, 0]
    
    # Convert to lists of ints as required
    edges = [list(e) for e in fano_edges]
    
    return {'edges': edges, 'colors': colors}
# EVOLVE-BLOCK-END
