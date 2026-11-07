"""Construct a 3-chromatic 3-uniform hypergraph on 9 vertices with minimum edges."""

def construct():
    # The Fano plane is a 3-uniform hypergraph on 7 vertices with 7 edges
    # that is 3-chromatic (cannot be colored with 2 colors, but can with 3).
    # We add 2 isolated vertices (7, 8) to reach 9 vertices.
    # 7 edges is the theoretical minimum for a 3-chromatic 3-uniform hypergraph.
    
    edges = [
        [0, 1, 2],
        [0, 3, 4],
        [0, 5, 6],
        [1, 3, 5],
        [1, 4, 6],
        [2, 3, 6],
        [2, 4, 5]
    ]
    
    # Verify this is a valid 3-coloring (no edge is monochromatic)
    colors = [0, 1, 2, 1, 2, 0, 1, 0, 0]
    
    # Double-check all edges are non-monochromatic
    for e in edges:
        c = {colors[e[0]], colors[e[1]], colors[e[2]]}
        if len(c) == 1:
            raise ValueError("Monochromatic edge detected")
    
    # Ensure all types are correct Python int and list
    edges = [[int(x) for x in e] for e in edges]
    colors = [int(c) for c in colors]
    
    return {'edges': edges, 'colors': colors}
