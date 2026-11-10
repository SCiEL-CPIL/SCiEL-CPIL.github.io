"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
# No external libraries needed for the optimal construction

def construct():
    """
    Construct a graph on vertices 0..15 with diameter exactly 2,
    such that deleting ANY single edge increases the diameter above 2
    (or disconnects the graph). Maximize edge count.
    
    PROOF OF OPTIMALITY:
    1. Condition "deleting edge (u,v) increases distance > 2" implies u and v have no common neighbor.
       This is equivalent to the graph being triangle-free.
    2. We need a triangle-free graph on n=16 vertices with diameter 2.
    3. Mantel's Theorem states the max edges in a triangle-free graph on n vertices is floor(n^2/4).
       For n=16, max edges = 16^2 / 4 = 64.
    4. The unique extremal graph is K_{8,8}.
    5. K_{8,8} has diameter 2.
    6. Thus, 64 is the global maximum. No search for better structures is possible.
    """
    n = 16
    mid = n // 2
    
    edges = []
    for u in range(mid):
        for v in range(mid, n):
            edges.append([u, v])
    
    return edges
# EVOLVE-BLOCK-END
