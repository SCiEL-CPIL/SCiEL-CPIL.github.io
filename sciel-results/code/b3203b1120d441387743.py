"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    """
    Construct a C5-free graph with no independent set of 9 vertices.
    
    Strategy: Use a disjoint union of 8 copies of K4 (complete graph on 4 vertices).
    - Total vertices: 8 * 4 = 32
    - C5-free: Each component is K4, which has no 5-cycle. Disjoint union preserves C5-freeness.
    - Independence number: alpha = 8 (pick at most 1 vertex from each K4 to be independent).
      Since 8 < 9, there is no independent set of size 9.
    - This certifies R(C5, K9) > 32.
    """
    n = 32
    edges = []
    for block in range(8):
        base = 4 * block
        for i in range(4):
            for j in range(i + 1, 4):
                edges.append([base + i, base + j])
    return {'n': n, 'edges': edges}
# EVOLVE-BLOCK-END
