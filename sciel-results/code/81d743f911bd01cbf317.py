"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    """
    Construct a dense graph on vertices 0..15 with diameter exactly two,
    where every edge is essential (deleting any edge increases diameter above 2
    or disconnects the graph). Maximize edge count.

    The condition that every edge is critical for diameter 2 requires that
    for every edge (u,v), there is no alternative path of length 2 between
    u and v. This is equivalent to N(u) ∩ N(v) = ∅ for all edges (u,v),
    which means the graph is triangle-free.

    By Mantel's theorem, the maximum number of edges in a triangle-free
    graph on 16 vertices is floor(16^2/4) = 64, achieved uniquely by
    K_{8,8}.

    In K_{8,8} with partitions A={0..7} and B={8..15}:
    - Diameter is 2 (vertices in same part have distance 2 via any vertex
      in the other part; vertices in different parts have distance 1).
    - For any edge (u,v) with u∈A, v∈B: N(u)=B, N(v)=A, so N(u)∩N(v)=∅.
      Removing (u,v) makes dist(u,v)=3, so diameter becomes 3 > 2.
    - Every edge is essential.

    K_{8,8} with 64 edges is optimal.
    """
    edges = []
    for u in range(8):
        for v in range(8, 16):
            edges.append([u, v])
    return edges
# EVOLVE-BLOCK-END
