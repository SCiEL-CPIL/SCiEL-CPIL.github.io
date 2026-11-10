"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    """
    Construct a simple undirected graph on vertices 0..15 with diameter exactly 2,
    such that deleting ANY individual edge increases the diameter above 2 or disconnects the graph.
    Maximize the number of edges.
    
    The optimal solution is the complete bipartite graph K_{8,8}, which has 64 edges.
    This is the maximum number of edges in a triangle-free graph on 16 vertices (Mantel's Theorem).
    Since any triangle would make an edge non-essential, the graph must be triangle-free.
    K_{8,8} has diameter 2, and removing any edge increases the diameter to 3.
    """
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin for 60s limit
    
    n = 16
    mid = n // 2  # 8
    
    edges = []
    # Construct K_{8,8}
    for u in range(mid):
        for v in range(mid, n):
            edges.append([u, v])
            
    # Check deadline (trivially passed)
    if time.time() > deadline:
        return edges
        
    return edges
# EVOLVE-BLOCK-END
