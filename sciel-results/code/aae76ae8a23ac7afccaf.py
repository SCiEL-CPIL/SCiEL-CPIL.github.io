"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    """
    Construct a maximally diameter-2 graph on 16 vertices where every edge is critical.
    
    The complete bipartite graph K_{8,8} has 64 edges and diameter 2.
    It is a known result that the maximum number of edges in a graph with diameter 2
    where every edge is critical is floor(n^2/4), achieved by K_{n/2, n/2}.
    
    For n=16, this is 64 edges.
    
    Verification of criticality:
    For any edge (u, v) with u in A, v in B:
    - In K_{8,8}, dist(u, v) = 1.
    - After removing (u, v), u and v have no common neighbors (since A and B are disjoint).
    - Thus dist(u, v) becomes 3.
    - The diameter increases from 2 to 3.
    
    This makes every edge critical.
    """
    n = 16
    edges = []
    # Generate edges for K_{8,8}
    # Vertices 0-7 in set A, 8-15 in set B
    for i in range(8):
        for j in range(8, 16):
            edges.append([i, j])
    
    # Ensure all values are Python int
    result = [[int(e[0]), int(e[1])] for e in edges]
    
    return result
# EVOLVE-BLOCK-END
