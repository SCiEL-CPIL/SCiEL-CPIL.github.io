"""
Rewritten program using a deterministic optimal solution (Fano Plane).

The Fano plane is the unique 3-uniform hypergraph on 7 vertices that is 
3-chromatic and minimal. It has 7 edges. 
It is a proven result that any 3-uniform hypergraph with fewer than 7 edges 
is 2-colorable. Thus, 7 is the global minimum for the number of edges in 
a 3-chromatic 3-uniform hypergraph.

The task specifies vertices 0..8. We use the Fano plane on vertices 0..6 
and leave vertices 7 and 8 isolated. Isolated vertices do not affect the 
chromatic number or the edge count.
"""

def construct():
    # Construct Fano plane edges on vertices 0-6
    # The Fano plane is the projective plane of order 2.
    # Vertices: 0, 1, 2, 3, 4, 5, 6
    # Edges are lines of the plane.
    # Standard representation of Fano plane lines:
    edges = [
        [0, 1, 2],
        [0, 3, 4],
        [0, 5, 6],
        [1, 3, 5],
        [1, 4, 6],
        [2, 3, 6],
        [2, 4, 5]
    ]
    
    # A valid 3-coloring for the Fano plane.
    # The Fano plane is not 2-colorable, so it requires 3 colors.
    # Coloring:
    # v0: 0
    # v1: 1
    # v2: 2
    # v3: 1
    # v4: 2
    # v5: 0
    # v6: 1
    colors = [0, 1, 2, 1, 2, 0, 1, 0, 0]
    
    # Verify 3-chromatic property:
    # 1. Check that the provided coloring is valid (no monochromatic edges)
    for e in edges:
        c0, c1, c2 = colors[e[0]], colors[e[1]], colors[e[2]]
        if c0 == c1 == c2:
            raise ValueError("Provided coloring is invalid (monochromatic edge found)")
            
    # 2. Exhaustively check that no 2-coloring exists for the 7 vertices
    # There are 2^7 = 128 possible binary colorings.
    # A hypergraph is 2-colorable if there exists a binary assignment 
    # such that no edge is monochromatic.
    # We check if ANY binary coloring fails to be monochromatic on all edges.
    # If all binary colorings have at least one monochromatic edge, it's not 2-colorable.
    
    is_2_colorable = False
    for mask in range(128):
        # Generate binary coloring from mask
        binary_colors = [(mask >> i) & 1 for i in range(7)]
        
        valid_2_coloring = True
        for e in edges:
            c0, c1, c2 = binary_colors[e[0]], binary_colors[e[1]], binary_colors[e[2]]
            if c0 == c1 == c2:
                valid_2_coloring = False
                break
        
        if valid_2_coloring:
            is_2_colorable = True
            break
            
    if is_2_colorable:
        raise ValueError("Fano plane construction error: Found a 2-coloring")

    # Ensure types are strict Python int and list
    edges_out = [[int(x) for x in e] for e in edges]
    colors_out = [int(c) for c in colors]
    
    return {'edges': edges_out, 'colors': colors_out}
