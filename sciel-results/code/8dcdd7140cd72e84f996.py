def construct():
    # Build a C5-free graph with alpha < 9 on as many vertices as possible.
    # Use 8 disjoint copies of K4 (complete graph on 4 vertices).
    # Each K4 has independence number 1, so total alpha = 8 < 9.
    # No 5-cycles exist because the graph is a union of cliques of size 4.
    # Total vertices: 8 * 4 = 32.
    #
    # Construction:
    # For block b in 0..7, vertices are 4*b, 4*b+1, 4*b+2, 4*b+3
    # All pairs within each block are connected (K4).
    # No edges between different blocks.
    n = 32
    edges = []
    for b in range(8):
        base = 4 * b
        for i in range(4):
            for j in range(i + 1, 4):
                edges.append([base + i, base + j])
    return {'n': n, 'edges': edges}
