def construct():
    # Fano plane edges on vertices 0..6, with vertices 7,8 isolated
    # Standard Fano plane representation
    edges = [
        [0, 1, 2],
        [0, 3, 4],
        [0, 5, 6],
        [1, 3, 5],
        [1, 4, 6],
        [2, 3, 6],
        [2, 4, 5]
    ]
    
    # Valid 3-coloring: vertices 0..6 use colors 0,1,2; vertices 7,8 get color 0
    # 0:0, 1:1, 2:1, 3:2, 4:2, 5:1, 6:2
    colors = [0, 1, 1, 2, 2, 1, 2, 0, 0]
    
    # Verify every edge is non-monochromatic
    for e in edges:
        c = [colors[v] for v in e]
        if len(set(c)) == 1:
            raise ValueError(f"Edge {e} is monochromatic")
    
    # Exhaustively verify no 2-coloring exists
    # Try all 2^7 = 128 colorings of vertices 0..6 (vertices 7,8 don't matter for edges)
    import itertools
    for assignment in itertools.product([0, 1], repeat=7):
        valid = True
        for e in edges:
            c = [assignment[v] for v in e]
            if len(set(c)) == 1:
                valid = False
                break
        if valid:
            raise ValueError(f"Found a 2-coloring: {assignment}")
    
    return {
        'edges': edges,
        'colors': colors
    }
