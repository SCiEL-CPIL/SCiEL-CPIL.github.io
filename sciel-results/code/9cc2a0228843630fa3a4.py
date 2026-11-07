"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time


def construct():
    """Return edges of K_{8,8} on vertices 0..15, which achieves the theoretical
    maximum of 64 edges for a diameter-2 graph where every edge is critical."""
    n = 16
    half = 8
    edges = []
    for u in range(half):
        for v in range(half, n):
            edges.append([int(u), int(v)])
    return edges
# EVOLVE-BLOCK-END
