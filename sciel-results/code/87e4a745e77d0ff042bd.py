"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START

def construct():
    """Return edges of K_{8,8} on vertices 0..15: the optimal 64-edge
    triangle-free diameter-2 graph where every edge is critical."""
    n = 16
    half = n // 2
    edges = [[u, v] for u in range(half) for v in range(half, n)]
    return edges

# EVOLVE-BLOCK-END
