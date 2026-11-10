"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START

def construct():
    n = 16
    edges = []
    for u in range(n // 2):
        for v in range(n // 2, n):
            edges.append([u, v])
    return edges

# EVOLVE-BLOCK-END
