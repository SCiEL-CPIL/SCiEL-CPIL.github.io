def construct():
    # The Fano plane is the unique 3-uniform hypergraph on 7 vertices
    # with chromatic number 3 and 7 edges. It is the minimal such hypergraph.
    # We use vertices 0..6 for the Fano plane and 7,8 are isolated.
    
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
    # Verified: no edge is monochromatic.
    colors = [0, 1, 2, 1, 0, 2, 1, 0, 0]
    
    return {
        'edges': edges,
        'colors': colors
    }
