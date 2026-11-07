"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START

def construct():
    """
    Construct the complete bipartite graph K_{8,8} on vertices 0..15.
    Partition A = {0, ..., 7}
    Partition B = {8, ..., 15}
    
    This graph has 64 edges.
    It is triangle-free (required for edge-critical diameter 2).
    Its diameter is 2.
    Removing any edge (u, v) with u in A, v in B removes the unique path of length 1.
    Since the graph is bipartite, there are no paths of length 2 between u and v 
    (neighbors of u are in B, neighbors of v are in A, disjoint sets).
    The next shortest path has length 3.
    Thus, the diameter increases to 3 upon removal of any edge.
    """
    n = 16
    half = n // 2
    edges = []
    for u in range(half):
        for v in range(half, n):
            edges.append([u, v])
    return edges

# EVOLVE-BLOCK-END
