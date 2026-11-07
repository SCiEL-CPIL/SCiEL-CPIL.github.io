"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    """
    Constructs a graph on 16 vertices (0..15) with diameter 2.
    The graph is edge-critical with respect to diameter 2.
    Strategy: Use the complete bipartite graph K_{8,8}.
    - K_{8,8} is triangle-free.
    - In a triangle-free graph of diameter 2, every edge (u,v) has no common neighbors.
    - Removing (u,v) leaves u and v at distance >= 3 (since no path of length 2 exists).
    - Thus, diameter increases to 3.
    - K_{8,8} has 64 edges, which is the maximum possible for a triangle-free graph on 16 vertices (Mantel's Theorem).
    - Any graph with triangles would have edges in triangles. An edge in a triangle has common neighbors.
      Removing it would leave a path of length 2, so diameter might stay 2 unless the graph disconnects.
      But an edge in a triangle cannot be a bridge. So such edges would not be critical.
      Therefore, the graph must be triangle-free to maximize edges while maintaining criticality.
    """
    n = 16
    mid = n // 2
    edges = []
    
    # Partition 1: 0..7
    # Partition 2: 8..15
    for u in range(mid):
        for v in range(mid, n):
            edges.append([u, v])
            
    return edges
# EVOLVE-BLOCK-END
