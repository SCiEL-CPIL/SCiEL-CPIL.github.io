"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

P = {'n': 16}

def construct():
    # The optimal graph for this task is the complete bipartite graph K_{8,8}.
    # It has 64 edges.
    # It is triangle-free, so removing any edge (u,v) increases the distance between u and v to 3 (via a path u-w-x-v where w is in the other partition and x is in the first? No, u in A, v in B. Path u->b->a->v. Length 3).
    # Thus the diameter increases from 2 to 3, satisfying the "above two" condition.
    # Any graph with more edges must contain a triangle (Mantel's Theorem limit for triangle-free is n^2/4 = 64).
    # Graphs with triangles tend to have redundant edges for diameter 2, making them fail the criticality check.
    
    n = P['n']
    half = n // 2
    partition_a = list(range(half))
    partition_b = list(range(half, n))
    
    edges = []
    for u in partition_a:
        for v in partition_b:
            edges.append([u, v])
            
    return edges
# EVOLVE-BLOCK-END
