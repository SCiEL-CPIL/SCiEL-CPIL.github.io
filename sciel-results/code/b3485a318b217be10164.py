"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    n = 16
    half = n // 2
    edges = []
    for u in range(half):
        for v in range(half, n):
            edges.append([int(u), int(v)])
    return edges
# EVOLVE-BLOCK-END
