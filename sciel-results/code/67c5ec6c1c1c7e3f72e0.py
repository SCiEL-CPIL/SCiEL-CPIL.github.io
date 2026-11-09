"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START

def construct():
    # The Fano plane is the minimal 3-uniform hypergraph with chromatic number 3.
    # It has 7 edges. The seed program had 10 edges.
    # Score = 10 / 7 ~= 1.42857.
    
    # Vertices are 0..8. Fano plane uses 0..6. 7 and 8 are isolated.
    edges = [
        [0, 1, 2],
        [0, 3, 4],
        [0, 5, 6],
        [1, 3, 5],
        [1, 4, 6],
        [2, 3, 6],
        [2, 4, 5]
    ]
    
    # A valid 3-coloring for the Fano plane (extended to 9 vertices)
    # c: 0->0, 1->1, 2->2, 3->1, 4->2, 5->0, 6->1, 7->0, 8->0
    colors = [0, 1, 2, 1, 2, 0, 1, 0, 0]
    
    # Ensure types are Python int and list
    edges = [[int(e[0]), int(e[1]), int(e[2])] for e in edges]
    colors = [int(c) for c in colors]
    
    return {'edges': edges, 'colors': colors}

# EVOLVE-BLOCK-END
