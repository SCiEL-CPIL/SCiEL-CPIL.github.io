"""Construct a C5-free graph with no independent set of size 9, maximizing n."""
# EVOLVE-BLOCK-START
def construct():
    """
    Returns a simple graph on n vertices (0..n-1) that is:
    - C5-free (no 5-cycle as a subgraph)
    - has no independent set of size 9
    
    Construction: 8 disjoint copies of K4.
    n = 32.
    alpha(G) = 8 (one vertex from each K4).
    K4 is C5-free, so the disjoint union is C5-free.
    """
    n = 32
    edges = []
    for b in range(8):
        base = b * 4
        for i in range(4):
            for j in range(i + 1, 4):
                edges.append([base + i, base + j])
    return {'n': n, 'edges': edges}
# EVOLVE-BLOCK-END
