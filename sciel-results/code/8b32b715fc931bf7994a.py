"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random

# EVOLVE-BLOCK-START
def construct():
    """
    Construct the Clebsch graph (folded 5-cube) on 16 vertices.
    It is triangle-free, 4-chromatic, and has 40 edges.
    Vertices are represented by integers 0..15 (5-bit strings with MSB 0).
    Two vertices u, v are adjacent if their Hamming distance is 1 or 4.
    """
    edges = []
    for i in range(16):
        for j in range(i + 1, 16):
            diff = i ^ j
            dist = bin(diff).count('1')
            if dist == 1 or dist == 4:
                edges.append([i, j])
    return edges
# EVOLVE-BLOCK-END
