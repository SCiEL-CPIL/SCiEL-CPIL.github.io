# EVOLVE-BLOCK-START
def construct():
    """
    Constructs a graph with n=32 vertices consisting of 8 disjoint copies of K4.
    - No 5-cycles (K4 is triangle-free? No, K4 has triangles, but no 5-cycles).
    - Independence number is 8 (one vertex from each K4).
    - This satisfies alpha(G) <= 8 and no C5.
    """
    n_blocks = 8
    block_size = 4
    n = n_blocks * block_size
    edges = []
    
    for b in range(n_blocks):
        base = b * block_size
        # Generate all edges for K4 on vertices base..base+3
        for i in range(block_size):
            for j in range(i + 1, block_size):
                edges.append([base + i, base + j])
                
    return {'n': n, 'edges': edges}
# EVOLVE-BLOCK-END
