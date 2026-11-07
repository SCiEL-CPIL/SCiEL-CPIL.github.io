"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START

def construct():
    n = 16
    edges = []
    # Partition vertices into two sets of 8
    # A = 0..7, B = 8..15
    for u in range(8):
        for v in range(8, 16):
            edges.append([u, v])
    return edges

# EVOLVE-BLOCK-END
