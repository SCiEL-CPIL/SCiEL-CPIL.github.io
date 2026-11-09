"""Construct a dense critical diameter-2 graph on 16 vertices."""
# EVOLVE-BLOCK-START

def construct():
    """
    Return edges of the complete bipartite graph K_{8,8} on vertices 0..15.
    
    Properties:
    - Diameter is exactly 2 (vertices in same partition have distance 2, 
      vertices in different partitions have distance 1).
    - Every edge is critical: deleting any edge (u,v) where u in A, v in B 
      makes u and v disconnected (since the graph is bipartite and complete,
      there are no other paths between A and B), so the graph becomes disconnected.
    - Edge count: 8 * 8 = 64, which is maximal for this class of graphs.
    
    The graph has no loops or duplicates.
    """
    n = 16
    half = n // 2
    edges = []
    for u in range(half):
        for v in range(half, n):
            edges.append([int(u), int(v)])
    return edges

# EVOLVE-BLOCK-END
