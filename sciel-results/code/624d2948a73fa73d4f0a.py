"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    """
    Returns a graph on 32 vertices with no C5 and no independent set of size 9.
    Construction: 8 disjoint copies of K4.
    - Each K4 has 4 vertices and 6 edges.
    - No C5 exists in K4, and disjoint unions preserve this.
    - Independence number of K4 is 1. Total independence number is 8, which is < 9.
    """
    n = 32
    edges = []
    for b in range(8):
        base = 4 * b
        for i in range(4):
            for j in range(i + 1, 4):
                edges.append([base + i, base + j])
    return {'n': n, 'edges': edges}
# EVOLVE-BLOCK-END
