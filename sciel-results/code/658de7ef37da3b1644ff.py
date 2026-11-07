"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START

def construct():
    n = 16
    edges = []
    # K_{8,8}: bipartition {0..7} and {8..15}
    for u in range(8):
        for v in range(8, 16):
            edges.append([u, v])
    return edges

# EVOLVE-BLOCK-END
