"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""

# EVOLVE-BLOCK-START

def construct():
    # The minimum number of edges in a 3-uniform hypergraph with chromatic number 3 is 7.
    # The Fano plane is the unique such hypergraph with 7 vertices.
    # We use vertices 0..6 for the Fano plane and leave 7, 8 isolated.
    
    edges = [
        [0, 1, 2],
        [0, 3, 4],
        [0, 5, 6],
        [1, 3, 5],
        [1, 4, 6],
        [2, 3, 6],
        [2, 4, 5]
    ]
    
    # A valid 3-coloring for the Fano plane
    # Vertices: 0, 1, 2, 3, 4, 5, 6, 7, 8
    # Colors:
    # 0: 0
    # 1: 1
    # 2: 2
    # 3: 1
    # 4: 2
    # 5: 0
    # 6: 1
    # 7: 0 (isolated)
    # 8: 0 (isolated)
    
    colors = [0, 1, 2, 1, 2, 0, 1, 0, 0]
    
    # Verify coloring is valid (no monochromatic edges)
    for e in edges:
        c0, c1, c2 = colors[e[0]], colors[e[1]], colors[e[2]]
        if c0 == c1 == c2:
            # This should never happen for the Fano plane with this coloring
            raise ValueError("Invalid coloring: monochromatic edge found")
            
    # Ensure types are correct
    edges_out = [list(map(int, e)) for e in edges]
    colors_out = list(map(int, colors))
    
    return {'edges': edges_out, 'colors': colors_out}

# EVOLVE-BLOCK-END
