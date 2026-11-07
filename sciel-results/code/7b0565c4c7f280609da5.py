# EVOLVE-BLOCK-START
def construct():
    n = 16
    # K_{8,8} is the unique extremal triangle-free graph on 16 vertices with 64 edges.
    # It has diameter 2 and every edge is essential.
    # Vertices 0-7 in one partition, 8-15 in the other.
    edges = []
    for i in range(8):
        for j in range(8):
            edges.append([i, j + 8])
    return edges
# EVOLVE-BLOCK-END
