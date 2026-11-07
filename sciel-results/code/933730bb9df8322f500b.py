"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""

# EVOLVE-BLOCK-START

def construct():
    # The Fano plane is the unique 3-uniform hypergraph with 7 edges
    # that is 3-chromatic. Any 3-uniform hypergraph with <= 6 edges is 2-colorable.
    # We construct the Fano plane on vertices 0..6. Vertices 7 and 8 are isolated.
    
    # Define the Fano plane edges
    # Standard Fano plane lines
    fano_lines = [
        (0, 1, 2),
        (0, 3, 4),
        (0, 5, 6),
        (1, 3, 5),
        (1, 4, 6),
        (2, 3, 6),
        (2, 4, 5)
    ]
    
    edges = [list(e) for e in fano_lines]
    
    # Construct a valid 3-coloring.
    # In the Fano plane, no edge is monochromatic in any 2-coloring.
    # A valid 3-coloring can be derived by assigning colors based on vertex indices modulo 3, 
    # but we must check if that works. 
    # Let's use the known valid coloring:
    # Vertices: 0, 1, 2, 3, 4, 5, 6
    # Colors:   0, 1, 2, 1, 2, 0, 1
    # Let's verify this programmatically to be safe.
    
    candidate_colors = [0, 1, 2, 1, 2, 0, 1, 0, 0]
    
    # Verification: Ensure every edge is non-monochromatic
    for e in edges:
        c0 = candidate_colors[e[0]]
        c1 = candidate_colors[e[1]]
        c2 = candidate_colors[e[2]]
        if c0 == c1 == c2:
            raise ValueError("Invalid coloring: found monochromatic edge")
            
    # Verification: Ensure the hypergraph is not 2-colorable.
    # Since the number of vertices is small (9), we can exhaustively check all 2^9 binary colorings.
    # A 2-coloring assigns 0 or 1 to each vertex. An edge is monochromatic if all three vertices have the same color.
    # If no 2-coloring exists where all edges are non-monochromatic, the hypergraph is 3-chromatic (or higher).
    
    n_vertices = 9
    is_2_colorable = False
    
    # Iterate through all 2^n_vertices possible colorings
    # We can use bit manipulation for speed, but 512 iterations is trivial.
    for mask in range(1 << n_vertices):
        # Extract the coloring for each vertex
        valid_2_coloring = True
        for e in edges:
            v0, v1, v2 = e
            # Get the bit for each vertex
            b0 = (mask >> v0) & 1
            b1 = (mask >> v1) & 1
            b2 = (mask >> v2) & 1
            
            # Check if monochromatic
            if b0 == b1 == b2:
                valid_2_coloring = False
                break
        
        if valid_2_coloring:
            is_2_colorable = True
            break
            
    if is_2_colorable:
        # This should not happen for the Fano plane, but if it does, the construction is wrong for the task requirements.
        # The task requires a 3-chromatic hypergraph.
        raise ValueError("Constructed hypergraph is 2-colorable, which is invalid for the task.")
        
    # Return the result with correct types
    return {
        'edges': edges,
        'colors': [int(c) for c in candidate_colors]
    }

# EVOLVE-BLOCK-END
