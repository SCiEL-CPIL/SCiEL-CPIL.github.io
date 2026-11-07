"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START

def construct():
    # The Fano plane is the unique minimal 3-uniform hypergraph with chromatic number 3.
    # It has 7 vertices and 7 edges. Property B (not 2-colorable) for 3-uniform
    # hypergraphs requires at least 7 edges, and the Fano plane achieves this bound.
    # We place the Fano plane on vertices 0..6 and leave vertices 7,8 isolated.
    # This gives a 3-chromatic hypergraph on 9 vertices with exactly 7 edges.
    
    edges = [
        [0, 1, 2],
        [0, 3, 4],
        [0, 5, 6],
        [1, 3, 5],
        [1, 4, 6],
        [2, 3, 6],
        [2, 4, 5]
    ]
    
    # Valid 3-coloring for the Fano plane:
    # 0:0, 1:1, 2:2, 3:0, 4:1, 5:2, 6:0
    # Vertices 7, 8 are isolated, assign color 0.
    colors = [0, 1, 2, 0, 1, 2, 0, 0, 0]
    
    # Verify: every edge must be non-monochromatic
    for e in edges:
        c0, c1, c2 = colors[e[0]], colors[e[1]], colors[e[2]]
        if c0 == c1 == c2:
            # This should never happen for the Fano plane with this coloring.
            # Fallback to a known valid configuration (combinations on 5 vertices).
            from itertools import combinations
            return {'edges': [list(e) for e in combinations(range(5), 3)],
                    'colors': [0, 0, 1, 1, 2, 0, 0, 0, 0]}
    
    # Ensure types are exactly Python int and list
    edges = [list(map(int, e)) for e in edges]
    colors = list(map(int, colors))
    
    return {'edges': edges, 'colors': colors}

# EVOLVE-BLOCK-END
