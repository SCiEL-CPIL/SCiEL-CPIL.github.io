"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    # The optimal graph is the complete bipartite graph K_{8,8}.
    # Vertices 0-7 are in partition A, vertices 8-15 are in partition B.
    # Edges exist between every u in A and v in B.
    # This graph has diameter 2.
    # Removing any edge (u, v) makes u and v non-adjacent.
    # Since u and v are in different partitions, they have no common neighbors 
    # in K_{8,8} (neighbors of u are in B, neighbors of v are in A).
    # Thus d(u, v) becomes 3, so the diameter increases to 3.
    
    # K_{8,8} is the unique optimal solution with 64 edges.
    # Using list comprehension for slight speed improvement.
    edges = [[u, v] for u in range(8) for v in range(8, 16)]
    return edges
# EVOLVE-BLOCK-END
