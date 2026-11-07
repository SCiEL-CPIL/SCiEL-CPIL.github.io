import time

P = {'n': 16}

def construct():
    """
    Returns a list of edges for the complete bipartite graph K_{8,8}.
    
    Reasoning:
    1. The condition that deleting any edge increases the diameter above 2 (or disconnects)
       implies that no edge can be part of a triangle, unless its removal disconnects the graph.
       In a dense graph maximizing edges, we assume high connectivity, so edges are not bridges.
       Thus, the graph must be triangle-free.
    2. By Mantel's Theorem, the maximum number of edges in a triangle-free graph on n vertices
       is floor(n^2 / 4).
    3. For n=16, max edges = 64, achieved by K_{8,8}.
    4. K_{8,8} has diameter 2.
    5. In K_{8,8}, any edge (u,v) connects partitions V1 and V2. N(u) subset V2, N(v) subset V1.
       N(u) intersect N(v) is empty.
       Removing (u,v) removes the direct link. No path of length 2 exists (requires common neighbor).
       Shortest path becomes length 3 (u -> w1 -> w2 -> v).
       Diameter increases from 2 to 3.
       Condition satisfied.
    """
    n = P['n']
    
    if n < 2:
        return []
    
    # Split vertices into two partitions as evenly as possible
    half = n // 2
    part1 = list(range(half))
    part2 = list(range(half, n))
    
    edges = []
    
    # Construct all cross edges
    for u in part1:
        for v in part2:
            edges.append([u, v])
            
    return edges
