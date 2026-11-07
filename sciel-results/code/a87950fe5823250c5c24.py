"""
Constructs a maximal diameter-2 critical graph on 16 vertices.
Theoretically, such graphs must be triangle-free.
By Mantel's Theorem, the maximum number of edges in a triangle-free graph
on n vertices is floor(n^2 / 4), achieved by the complete bipartite graph
K_{n//2, n-n//2}.
For n=16, this is K_{8,8} with 64 edges.
"""

P = {'n': 16}

def construct():
    """
    Returns a list of edges for the complete bipartite graph K_{8,8}.
    This graph has diameter 2.
    It is triangle-free, which is a necessary condition for all edges
    to be critical (deleting an edge in a triangle doesn't change diameter).
    In K_{8,8}, deleting any edge (u,v) where u in Part1, v in Part2
    results in d(u,v) becoming 3 (odd distance in bipartite graph),
    thus increasing the diameter from 2 to 3.
    Hence all edges are critical.
    """
    n = P['n']
    
    if n < 2:
        return []
    
    # Split vertices into two partitions
    half = n // 2
    part1 = [int(i) for i in range(half)]
    part2 = [int(i) for i in range(half, n)]
    
    edges = []
    
    # Construct all cross edges
    for u in part1:
        for v in part2:
            edges.append([int(u), int(v)])
            
    return edges
