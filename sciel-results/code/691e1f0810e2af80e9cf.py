"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

P = {'n': 16}

def construct():
    """
    Returns the edge list of K_{8,8}, the unique (up to isomorphism) triangle-free
    graph on 16 vertices with the maximum number of edges (64) that has diameter 2
    and where every edge is essential.
    
    Proof of optimality:
    - For every edge (u,v) to be essential, u and v must have no common neighbor.
    - This implies the graph is triangle-free.
    - By Turán's theorem, max edges in triangle-free graph on n vertices is floor(n^2/4).
    - For n=16: floor(256/4) = 64, achieved uniquely by K_{8,8}.
    - K_{8,8} has diameter 2 (any two vertices in different partitions are adjacent;
      any two in the same partition have a common neighbor in the other partition).
    - Every edge is essential: removing (u,v) with u in A, v in B leaves no path of
      length <=2 between u and v (would require a vertex in both A and B).
    """
    n = P['n']
    mid = n // 2
    
    edges = [[int(u), int(v)] for u in range(0, mid) for v in range(mid, n)]
    return edges
# EVOLVE-BLOCK-END
