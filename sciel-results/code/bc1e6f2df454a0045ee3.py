"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START

def construct():
    n = 16
    half = n // 2
    edges = []
    # Construct complete bipartite graph K_{8,8}
    # Partitions: 0..7 and 8..15
    for u in range(half):
        for v in range(half, n):
            edges.append([u, v])
    return edges

# EVOLVE-BLOCK-END
