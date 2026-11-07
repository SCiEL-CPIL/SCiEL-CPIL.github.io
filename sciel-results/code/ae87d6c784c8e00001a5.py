def construct():
    """
    Constructs a maximal edge-critical diameter-2 graph on 16 vertices.
    
    The optimal graph is the complete bipartite graph K_{8,8}.
    - Vertices are partitioned into two sets A and B of size 8 each.
    - All edges connect a vertex in A to a vertex in B.
    - Total edges: 8 * 8 = 64.
    - Diameter is 2.
    - Edge-critical: Removing any edge (u, v) where u in A and v in B
      leaves u and v with no common neighbors, so dist(u,v) becomes 3.
    """
    n = 16
    half = n // 2

    edges = []
    for u in range(half):
        for v in range(half, n):
            edges.append([int(u), int(v)])

    return edges
