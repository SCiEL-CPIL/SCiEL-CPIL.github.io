"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
P = {'n': 16}

def construct():
    n = P['n']
    # Split vertices into two sets of size n/2
    # Vertices 0..7 in L, 8..15 in R
    edges = []
    for u in range(n // 2):
        for v in range(n // 2, n):
            edges.append([u, v])
    return edges
# EVOLVE-BLOCK-END
