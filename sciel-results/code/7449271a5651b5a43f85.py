"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START

def construct():
    # The Fano plane is the smallest 3-uniform hypergraph that is not 2-colorable.
    # It has 7 edges and is 3-chromatic.
    # We embed it in 9 vertices (0..8) by using vertices 0..6 and leaving 7,8 isolated.
    edges = [
        [0, 1, 2],
        [0, 3, 4],
        [0, 5, 6],
        [1, 3, 5],
        [1, 4, 6],
        [2, 3, 6],
        [2, 4, 5]
    ]
    
    # A valid 3-coloring for the Fano plane vertices 0..6
    # c(0)=0, c(1)=1, c(2)=2, c(3)=1, c(4)=0, c(5)=0, c(6)=1
    # Vertices 7, 8 can be any color, e.g., 0
    colors = [0, 1, 2, 1, 0, 0, 1, 0, 0]
    
    return {'edges': edges, 'colors': colors}

# EVOLVE-BLOCK-END
