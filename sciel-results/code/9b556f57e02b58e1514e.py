# EVOLVE-BLOCK-START

def construct():
    """
    Constructs a simple undirected graph on 16 vertices (0..15) with diameter exactly 2,
    where every edge is essential (critical).
    
    The optimal such graph is the complete bipartite graph K_{8,8}.
    - It is triangle-free. In a triangle-free graph with diameter 2, every edge (u,v) 
      has no common neighbor w. Thus, removing (u,v) increases d(u,v) from 1 to 3.
    - It has the maximum number of edges among triangle-free graphs on 16 vertices 
      (Mantel's Theorem: floor(16^2 / 4) = 64).
    
    Returns:
        List of edges [u, v] where u and v are Python ints.
    """
    n = 16
    half = n // 2
    
    # Partition vertices into two sets: 0..7 and 8..15
    edges = [[u, v] for u in range(half) for v in range(half, n)]
    return edges

# EVOLVE-BLOCK-END
