"""
Constructs K_{8,8}, the unique triangle-free graph on 16 vertices 
with maximum edge count (64) and diameter exactly 2.

Every edge is critical: 
1. The graph is triangle-free (bipartite), so removing any edge (u, v) 
   eliminates the only path of length 1 between u and v, and no path 
   of length 2 exists (no common neighbor). 
2. Thus, dist(u, v) becomes >= 3 (or infinity if disconnected, though 
   K_{8,8} minus one edge remains connected), increasing the diameter 
   above 2.

By Mantel's Theorem, no triangle-free graph on 16 vertices can have 
more than 64 edges. Any graph with more edges must contain a triangle, 
which implies the existence of non-critical edges (edges in a triangle 
have an alternative path of length 2, so their removal does not 
necessarily increase the diameter above 2).
"""

def construct():
    """
    Returns the edges of K_{8,8}.
    """
    n = 16
    half = n // 2
    edges = []
    
    # Generate edges for complete bipartite graph K_{8,8}
    # Partition 1: 0..7
    # Partition 2: 8..15
    for u in range(half):
        for v in range(half, n):
            edges.append([int(u), int(v)])
            
    return edges
