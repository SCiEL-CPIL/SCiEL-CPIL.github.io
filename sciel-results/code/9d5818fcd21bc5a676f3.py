"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    """
    Construct a C5-free graph on 32 vertices with no independent set of size 9.
    
    Approach: Disjoint union of 8 copies of K4.
    - Each K4 is a clique of size 4, which is C5-free (no cycles at all).
    - The independence number of K4 is 1 (all vertices are mutually adjacent).
    - For 8 disjoint K4s:
        * Total vertices n = 8 * 4 = 32
        * Total independence number alpha = 8 * 1 = 8 < 9
        * No C5 exists since each component is a clique (and cliques of size <= 4 have no cycles)
    
    This achieves n = 32, which matches the best known construction.
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
