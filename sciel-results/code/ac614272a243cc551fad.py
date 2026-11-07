"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START

def construct():
    """
    Return a 3-uniform hypergraph on 9 vertices (0..8) that is 3-chromatic
    (not 2-colorable).
    
    Strategy: Use the 7 edges of the Fano plane on vertices 0-6, plus one
    additional edge {0, 7, 8} to involve all 9 vertices.
    Total edges: 8.
    """
    edges = [
        [0, 1, 2],
        [0, 3, 4],
        [0, 5, 6],
        [1, 3, 5],
        [1, 4, 6],
        [2, 3, 6],
        [2, 4, 5],
        [0, 7, 8]
    ]
    
    # Verify no 2-coloring exists (brute force over 2^9 = 512 assignments)
    from itertools import product
    for coloring in product([0, 1], repeat=9):
        valid = True
        for edge in edges:
            if len(set(coloring[v] for v in edge)) == 1:
                valid = False
                break
        if valid:
            raise ValueError("Found a 2-coloring, hypergraph should not be 2-colorable")
    
    # A valid 3-coloring
    # Fano plane coloring: 0:0, 1:1, 2:2, 3:1, 4:0, 5:2, 6:1
    # Edge {0,7,8} needs non-monochromatic colors.
    # Let's assign 7:0, 8:1.
    # Edge {0,7,8} colors: 0,0,1 -> non-monochromatic.
    colors = [0, 1, 2, 1, 0, 2, 1, 0, 1]
    
    # Verify the provided coloring is valid
    for edge in edges:
        if len(set(colors[v] for v in edge)) == 1:
            raise ValueError("Provided coloring is not valid")
    
    return {'edges': edges, 'colors': colors}

# EVOLVE-BLOCK-END
