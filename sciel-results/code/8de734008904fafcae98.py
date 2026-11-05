"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    # Eight disjoint copies of K4: no 5-cycle and independence number 8.
    # n = 8 * 4 = 32
    # Edges: for each block b in 0..7, vertices 4b to 4b+3 form a K4.
    edges = []
    for b in range(8):
        base = 4 * b
        for i in range(4):
            for j in range(i + 1, 4):
                edges.append([base + i, base + j])
    return {'n': 32, 'edges': edges}
# EVOLVE-BLOCK-END
