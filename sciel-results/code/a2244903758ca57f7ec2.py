"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from collections import deque

def construct():
    """
    Construct a graph on vertices 0..15 with diameter exactly 2,
    such that deleting ANY single edge increases the diameter above 2
    (or disconnects the graph). Maximize edge count.
    
    Key insight: A complete bipartite graph K_{8,8} has 64 edges, diameter 2,
    and every edge is essential (removing any edge (u,v) makes dist(u,v)=3).
    
    But can we do better? Consider adding edges within partitions.
    If we add an edge within partition A, say (a1, a2), then for any v in B,
    removing edge (a1, v) still leaves path a1-a2-v of length 2, so (a1,v) is NOT essential.
    So we cannot add intra-partition edges if we want all edges to be essential.
    
    Therefore K_{8,8} with 64 edges seems optimal for this structure.
    
    However, maybe a different structure allows more edges?
    
    Let's think: we need a graph where for every edge (u,v), there is NO other path of length <= 2 between u and v.
    This means: for every edge (u,v), u and v must not share a common neighbor, AND there must be no edge (u,v) alternative... wait, (u,v) is the only direct edge.
    Actually: after removing (u,v), the distance must be > 2. So there must be no path of length 1 (impossible, we removed it) or length 2.
    No path of length 2 means: u and v have NO common neighbor.
    
    So the condition is: the graph is triangle-free, AND every edge belongs to no triangle.
    Wait, triangle-free means no 3-cycles, which means no edge has its endpoints sharing a neighbor.
    So the condition "removing edge (u,v) increases distance to > 2" is equivalent to: u and v have no common neighbor.
    Which is equivalent to: the graph is triangle-free.
    
    But we also need diameter exactly 2.
    
    So we need a triangle-free graph on 16 vertices with diameter 2, maximizing edges.
    By Mantel's theorem, the maximum triangle-free graph on n vertices is the complete bipartite graph K_{n/2, n/2}, with n^2/4 edges.
    For n=16, that's 64 edges.
    
    And K_{8,8} has diameter 2. So 64 is the maximum.
    """
    n = 16
    mid = n // 2
    
    edges = []
    for u in range(mid):
        for v in range(mid, n):
            edges.append([u, v])
    
    return edges
# EVOLVE-BLOCK-END
