def construct():
    """
    Return a 3-uniform hypergraph on vertices 0..8 that is 3-chromatic
    (not 2-colorable, but 3-colorable) with the minimum number of edges.
    
    The Fano plane is the unique (up to isomorphism) 3-chromatic 3-uniform
    hypergraph with 7 edges. It is known to be the minimum for 3-chromatic
    triple systems. We embed it in 9 vertices by leaving vertices 7 and 8 isolated.
    """
    # Fano plane edges (7 edges on vertices 0-6)
    edges = [
        [0, 1, 2],
        [0, 3, 4],
        [0, 5, 6],
        [1, 3, 5],
        [1, 4, 6],
        [2, 3, 6],
        [2, 4, 5]
    ]
    
    # A valid 3-coloring: no edge is monochromatic
    # Vertices 0-6 are colored to break each Fano edge
    # v0=0, v1=1, v2=2 -> edge [0,1,2] has colors {0,1,2} ✓
    # v0=0, v3=1, v4=0 -> edge [0,3,4] has colors {0,1} ✓
    # v0=0, v5=2, v6=0 -> edge [0,5,6] has colors {0,2} ✓
    # v1=1, v3=1, v5=2 -> edge [1,3,5] has colors {1,2} ✓
    # v1=1, v4=0, v6=0 -> edge [1,4,6] has colors {0,1} ✓
    # v2=2, v3=1, v6=0 -> edge [2,3,6] has colors {0,1,2} ✓
    # v2=2, v4=0, v5=2 -> edge [2,4,5] has colors {0,2} ✓
    colors = [0, 1, 2, 1, 0, 2, 0, 0, 0]
    
    # Ensure all types are Python int and list
    edges = [[int(a), int(b), int(c)] for a, b, c in edges]
    colors = [int(c) for c in colors]
    
    return {'edges': edges, 'colors': colors}
