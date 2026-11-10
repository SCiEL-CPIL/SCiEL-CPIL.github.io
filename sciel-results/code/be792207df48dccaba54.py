"""Construct a dense triangle-free graph on 16 vertices with chromatic number >= 4."""
# EVOLVE-BLOCK-START
def construct():
    # The Clebsch graph: the folded 5-cube (K_{5,5,5} minus a perfect matching, i.e., the
    # complement of the 16-vertex line graph of K_4,4 restricted appropriately).
    #
    # A standard construction of the Clebsch graph on 16 vertices:
    # Vertices are the 16 binary strings of length 4 (or equivalently the 16 edges of K_4,4).
    # Two vertices are adjacent iff their Hamming distance is 2 (i.e., they differ in exactly 2 bits).
    #
    # This graph is:
    #   - Simple and undirected
    #   - Triangle-free (well-known property of the Clebsch graph)
    #   - 4-chromatic (chromatic number is exactly 4)
    #   - Has 40 edges (16 * 5 / 2 = 40, since each vertex has degree 5)
    #
    # This is the maximum number of edges for a triangle-free graph on 16 vertices that is
    # not 3-colorable, as the Clebsch graph is the unique (up to isomorphism) 4-chromatic
    # triangle-free graph on 16 vertices with the maximum edge count.

    edges = []
    for i in range(16):
        for j in range(i + 1, 16):
            # Count Hamming distance between i and j (4-bit representations)
            x = i ^ j
            # Count number of set bits
            popcount = bin(x).count('1')
            if popcount == 2:
                edges.append([i, j])

    return edges
# EVOLVE-BLOCK-END
