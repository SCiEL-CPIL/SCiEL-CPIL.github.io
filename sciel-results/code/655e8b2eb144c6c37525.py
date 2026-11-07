"""Rewritten program using a deterministic optimal solution (Fano Plane)."""

def construct():
    # The Fano plane is the unique 3-uniform hypergraph on 7 vertices 
    # that is 3-chromatic and minimal (no proper sub-hypergraph is 3-chromatic).
    # It has 7 edges. Any 3-uniform hypergraph with fewer than 7 edges 
    # is 2-colorable.
    # We use vertices 0-6 for the Fano plane and leave 7, 8 isolated.
    
    edges = [
        (0, 1, 2),
        (0, 3, 4),
        (0, 5, 6),
        (1, 3, 5),
        (1, 4, 6),
        (2, 3, 6),
        (2, 4, 5)
    ]
    
    # A valid 3-coloring for the Fano plane structure:
    # v0=0, v1=1, v2=2
    # v3=1, v4=2, v5=0, v6=1
    # v7=0, v8=0 (isolated, can be any color)
    colors = [0, 1, 2, 1, 2, 0, 1, 0, 0]
    
    # Verify that the provided coloring is valid (no monochromatic edges)
    for e in edges:
        c0, c1, c2 = colors[e[0]], colors[e[1]], colors[e[2]]
        if c0 == c1 == c2:
            raise ValueError("Invalid coloring provided: monochromatic edge detected")

    # Return in required format
    return {
        'edges': [list(e) for e in edges],
        'colors': colors
    }
