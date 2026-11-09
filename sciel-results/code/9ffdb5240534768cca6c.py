"""Constructs a C5-free graph with independence number < 9."""
# EVOLVE-BLOCK-START
def construct():
    """
    Returns a graph on 32 vertices (8 disjoint copies of K4) which is C5-free
    and has independence number 8 (hence no independent set of size 9).
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
