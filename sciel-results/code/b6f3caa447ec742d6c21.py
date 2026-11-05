"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    # Eight disjoint copies of K4: no 5-cycle and independence number 8.
    # This achieves n=32, which is the maximum possible since R(C5, K9)=33.
    n = 32
    edges = []
    for block in range(8):
        base = 4 * block
        for i in range(4):
            for j in range(i + 1, 4):
                edges.append([base + i, base + j])
    return {'n': n, 'edges': edges}
# EVOLVE-BLOCK-END
