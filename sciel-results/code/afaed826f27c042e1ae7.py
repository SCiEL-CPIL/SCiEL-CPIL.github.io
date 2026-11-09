"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START

def construct():
    # The Fano plane is the unique 3-uniform hypergraph on 7 vertices
    # with 7 edges that is 3-chromatic (not 2-colorable).
    # We embed it on vertices 0-6. Vertices 7 and 8 are isolated.
    # This yields the minimum possible edge count of 7 for a 3-chromatic
    # 3-uniform hypergraph.
    
    edges = [
        [0, 1, 2],
        [0, 3, 4],
        [0, 5, 6],
        [1, 3, 5],
        [1, 4, 6],
        [2, 3, 6],
        [2, 4, 5]
    ]
    
    # A valid 3-coloring for the Fano plane vertices 0-6:
    # c(0)=0, c(1)=0, c(2)=1, c(3)=1, c(4)=2, c(5)=2, c(6)=0
    # Vertices 7 and 8 are assigned color 0.
    colors = [0, 0, 1, 1, 2, 2, 0, 0, 0]
    
    # Verify types are Python ints and lists
    edges = [[int(v) for v in e] for e in edges]
    colors = [int(c) for c in colors]
    
    return {'edges': edges, 'colors': colors}

# EVOLVE-BLOCK-END
