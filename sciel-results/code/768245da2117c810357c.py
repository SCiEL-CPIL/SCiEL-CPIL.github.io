"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
P = {'n': 16}

def construct():
    n = P['n']
    half = n // 2
    edges = []
    for i in range(half):
        for j in range(half, n):
            edges.append([i, j])
    return edges
# EVOLVE-BLOCK-END
