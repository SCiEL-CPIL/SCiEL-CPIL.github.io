"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
P = {'n': 16}

def construct():
    n = P['n']
    half = n // 2
    left = list(range(half))
    right = list(range(half, n))
    edges = []
    for u in left:
        for v in right:
            edges.append([u, v])
    return edges
# EVOLVE-BLOCK-END
