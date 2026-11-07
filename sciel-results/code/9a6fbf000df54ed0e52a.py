"""Rewritten program: find a 3-chromatic 3-uniform hypergraph on 9 vertices with minimum edges."""
# EVOLVE-BLOCK-START

def construct():
    """
    Returns a 3-chromatic 3-uniform hypergraph on vertices 0..8 with the minimum number of edges.
    We generate the Fano plane (the unique 3-chromatic 3-uniform hypergraph with 7 edges)
    on vertices 0..6 using their representation as non-zero vectors in GF(2)^3.
    Vertices 7 and 8 are isolated.
    """
    edges = []
    seen = set()
    
    # Generate edges based on XOR property of Fano plane points (1..7)
    for a in range(1, 8):
        for b in range(a + 1, 8):
            c = a ^ b
            if c == a or c == b:
                continue
            if c < 1 or c > 7:
                continue
            edge = tuple(sorted([a, b, c]))
            if edge not in seen:
                seen.add(edge)
                edges.append([a - 1, b - 1, c - 1])
                
    edges.sort()
    
    # Valid 3-coloring for this specific vertex labeling
    colors = [0, 1, 2, 0, 1, 2, 0, 0, 0]
    
    return {
        'edges': edges,
        'colors': colors
    }

# EVOLVE-BLOCK-END
