"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    """
    Constructs a graph on 32 vertices which is C5-free and has no independent set of size 9.
    The graph consists of 8 disjoint copies of K4.
    
    - Vertices: 0..31
    - Edges: All pairs within each block of 4 vertices.
    - C5-free: K4 has no cycles of length > 3. Disjoint union of C5-free graphs is C5-free.
    - Independence number: In each K4, the max independent set size is 1. 
      Total independent set size is sum of max independent sets of components = 8 * 1 = 8.
      8 < 9, so no independent set of size 9.
    """
    n = 32
    edges = []
    # 8 blocks of 4 vertices each
    for b in range(8):
        base = b * 4
        # Generate all edges within this K4
        for i in range(4):
            for j in range(i + 1, 4):
                edges.append([base + i, base + j])
    
    return {'n': n, 'edges': edges}
# EVOLVE-BLOCK-END
