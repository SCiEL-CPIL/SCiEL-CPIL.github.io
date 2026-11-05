"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    # Eight disjoint copies of K4: C5-free and independence number exactly 8.
    edges = []
    for b in range(8):
        for i in range(4):
            for j in range(i + 1, 4):
                u = 4 * b + i
                v = 4 * b + j
                edges.append([u, v])
    return {'n': 32, 'edges': edges}
# EVOLVE-BLOCK-END
