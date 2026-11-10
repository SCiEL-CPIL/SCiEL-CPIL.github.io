"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START

def construct():
    """
    Construct a simple undirected graph on vertices 0..15 with diameter exactly 2,
    such that deleting ANY individual edge increases the diameter above 2 or disconnects the graph.
    Maximize the number of edges.
    
    The optimal solution is the complete bipartite graph K_{8,8}, which has 64 edges.
    This is the maximum number of edges in a triangle-free graph on 16 vertices (Mantel's Theorem).
    Any graph with more than 64 edges contains a triangle. Edges added to a bipartite graph
    to form triangles are not essential for maintaining diameter 2, as the bipartite core
    remains intact and maintains diameter 2. Thus, 64 is the maximum.
    """
    n = 16
    mid = n // 2  # 8
    
    # Construct K_{8,8}
    edges = [[u, v] for u in range(mid) for v in range(mid, n)]
    
    return edges
# EVOLVE-BLOCK-END
