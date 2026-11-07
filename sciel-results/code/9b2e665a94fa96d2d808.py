"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    n = 16
    # Partition vertices into two sets of 8
    A = list(range(8))      # 0..7
    B = list(range(8, 16))  # 8..15
    # Complete bipartite graph K_{8,8}
    edges = []
    for a in A:
        for b in B:
            edges.append([a, b])
    return edges
# EVOLVE-BLOCK-END
