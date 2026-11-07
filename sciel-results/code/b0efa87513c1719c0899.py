"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
P = {'n': 15}

def construct():
    n = P['n']
    # Partition: A = 0..6, B = 7..14
    # Base: Complete Bipartite K_{7,8}
    edges = []
    for u in range(7):
        for v in range(7, 15):
            edges.append([u, v])
    # Add one edge within A: (0, 1)
    edges.append([0, 1])
    return edges
# EVOLVE-BLOCK-END
