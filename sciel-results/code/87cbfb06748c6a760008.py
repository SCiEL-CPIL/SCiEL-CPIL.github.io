def construct():
    """
    Constructs a simple undirected graph on vertices 0..15 with diameter 2
    such that deleting any single edge increases the diameter to at least 3.
    
    The graph chosen is the complete bipartite graph K_{8,8}.
    - Number of vertices: 16
    - Number of edges: 8 * 8 = 64
    - Diameter: 2 (any two nodes in the same partition are distance 2 via the other partition; nodes in different partitions are distance 1)
    - Edge Criticality: For any edge (u, v) with u in A and v in B, there is no path of length 2 between u and v other than the edge itself (because any intermediate node would need to be in both A and B). Thus, removing the edge increases the distance between u and v to 3 (via u -> w_in_B -> z_in_A -> v). Consequently, the diameter increases to 3.
    
    This is the maximum possible number of edges for a triangle-free graph on 16 vertices (Mantel's Theorem).
    """
    n = 16
    half = n // 2
    A = list(range(half))
    B = list(range(half, n))
    
    edges = []
    for u in A:
        for v in B:
            edges.append([u, v])
            
    return edges
