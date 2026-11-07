"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    n = 16
    # Complete bipartite graph K_{8,8} on vertices 0..15
    # Part A: 0..7, Part B: 8..15
    # Every edge is critical: removing (u,v) with u in A, v in B
    # makes d(u,v) = 3 (no common neighbor), so diameter becomes 3.
    # Graph has diameter 2.
    edges = []
    for u in range(8):
        for v in range(8, 16):
            edges.append([int(u), int(v)])
    return edges
# EVOLVE-BLOCK-END
