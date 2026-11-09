"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    """
    Constructs a graph on 32 vertices consisting of 8 copies of K4 connected in a path.
    - Each K4 has 4 vertices and 6 edges.
    - K4 does not contain a 5-cycle (C5).
    - Connecting K4s with single edges does not create C5s because any cycle 
      must stay within a single K4 (as crossing the bridge requires going back).
    - The independence number of a path of K4s is 8 (one from each K4).
    - n=32, alpha=8.
    """
    n = 32
    edges = []
    # 8 blocks of 4 vertices
    for b in range(8):
        base = 4 * b
        # Add edges for K4
        for i in range(4):
            for j in range(i + 1, 4):
                edges.append([base + i, base + j])
        # Connect to next block if not the last one
        if b < 7:
            # Connect vertex 0 of current block to vertex 0 of next block
            edges.append([base, base + 4])
    return {'n': n, 'edges': edges}
# EVOLVE-BLOCK-END
