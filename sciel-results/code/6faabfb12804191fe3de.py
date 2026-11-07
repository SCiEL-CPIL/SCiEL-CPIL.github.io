"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    """
    Return the edges of K_{8,8} on vertices 0..15.
    K_{8,8} is the unique (up to isomorphism) edge-maximal graph on 16 vertices
    with diameter exactly 2 and every edge critical (removing any edge increases
    diameter to 3).
    
    Partition: A = {0,1,...,7}, B = {8,9,...,15}
    Edges: all pairs (a, b) with a in A, b in B.
    Total edges: 8 * 8 = 64.
    """
    n = 16
    mid = n // 2  # 8
    edges = []
    for u in range(mid):
        for v in range(mid, n):
            edges.append([int(u), int(v)])
    return edges
# EVOLVE-BLOCK-END
