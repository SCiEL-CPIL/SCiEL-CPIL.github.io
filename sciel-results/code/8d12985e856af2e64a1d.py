def construct():
    # The Fano plane is the minimal 3-uniform hypergraph that is not 2-colorable.
    # It has 7 edges. We embed it in 9 vertices by adding isolated vertices 7 and 8.
    # This configuration is 3-chromatic (requires 3 colors, not 2).
    
    # Verified valid edges for the Fano plane on vertices 0-6
    # This specific labeling is known to work with the coloring below.
    edges = [
        [0, 1, 2],
        [0, 3, 4],
        [0, 5, 6],
        [1, 3, 5],
        [1, 4, 6],
        [2, 3, 6],
        [2, 4, 5]
    ]
    
    # 3-coloring for vertices 0-8
    # Vertices 7 and 8 are isolated, so their color is arbitrary (set to 0).
    colors = [0, 1, 2, 0, 1, 2, 0, 0, 0]
    
    # Verification Step 1: Check that the provided 3-coloring is valid
    # (i.e., no edge is monochromatic)
    for e in edges:
        c0, c1, c2 = colors[e[0]], colors[e[1]], colors[e[2]]
        if c0 == c1 == c2:
            raise ValueError("Provided 3-coloring is invalid: monochromatic edge found")
            
    # Verification Step 2: Check that no 2-coloring exists
    # We exhaustively check all 2^9 possible 2-colorings.
    # A 2-coloring is valid if NO edge is monochromatic.
    # If we find ANY valid 2-coloring, the hypergraph is 2-colorable (failure).
    n_vertices = 9
    for mask in range(1 << n_vertices):
        is_valid_2_coloring = True
        for e in edges:
            c0 = (mask >> e[0]) & 1
            c1 = (mask >> e[1]) & 1
            c2 = (mask >> e[2]) & 1
            if c0 == c1 == c2:
                is_valid_2_coloring = False
                break
        if is_valid_2_coloring:
            raise ValueError("Hypergraph is 2-colorable! Found a valid 2-coloring.")

    # Ensure types are exactly Python int
    final_edges = []
    for e in edges:
        final_edges.append([int(v) for v in e])
        
    final_colors = [int(c) for c in colors]
    
    return {
        'edges': final_edges,
        'colors': final_colors
    }
