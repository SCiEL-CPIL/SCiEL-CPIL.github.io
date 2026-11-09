"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    # Fano plane edges on vertices 0-6
    edges = [
        [0, 1, 2],
        [0, 3, 4],
        [0, 5, 6],
        [1, 3, 5],
        [1, 4, 6],
        [2, 3, 6],
        [2, 4, 5]
    ]
    # Valid 3-coloring for Fano plane vertices 0-6
    # c(0)=0, c(1)=1, c(2)=2, c(3)=0, c(4)=1, c(5)=0, c(6)=2
    colors_fano = [0, 1, 2, 0, 1, 0, 2]
    # Vertices 7 and 8 are isolated, color them 0
    colors = colors_fano + [0, 0]
    
    return {'edges': edges, 'colors': colors}
# EVOLVE-BLOCK-END
