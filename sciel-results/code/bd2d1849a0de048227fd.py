"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START

def construct():
    """
    Constructs the Fano plane (PG(2,2)) as a 3-uniform hypergraph on 9 vertices.
    The 7 lines of the Fano plane are the proven minimum for a 3-chromatic
    3-uniform hypergraph. Vertices 7 and 8 are isolated.
    """
    # Fano plane edges: for each pair (i,j) with 1<=i<j<=7, k=i^j gives the third point
    # on the line. We collect unique triples where i<j<k.
    edges = []
    for i in range(1, 8):
        for j in range(i + 1, 8):
            k = i ^ j
            if k > j:
                edges.append([i - 1, j - 1, k - 1])
    edges.sort()

    # Valid 3-coloring: c(v) = v % 3 for v in 0..6, extended to 9 vertices.
    # Verified: no Fano plane line is monochromatic under this assignment.
    colors = [v % 3 for v in range(9)]

    return {
        'edges': [[int(e[0]), int(e[1]), int(e[2])] for e in edges],
        'colors': [int(c) for c in colors]
    }

# EVOLVE-BLOCK-END
