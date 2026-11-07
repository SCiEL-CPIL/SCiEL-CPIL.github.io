"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    n = 15
    # Complete bipartite graph K_{7,8}
    # Part A: vertices 0-6 (7 vertices, each degree 8)
    # Part B: vertices 7-14 (8 vertices, each degree 7)
    # Any 3-edge path s-u-v-t has s and t in the same part,
    # so deg(s) != deg(t) since parts have different uniform degrees.
    edges = []
    for a in range(7):
        for b in range(7, 15):
            edges.append([a, b])
    return edges
# EVOLVE-BLOCK-END
