"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    n = 16
    deadline = time.time() + 80
    # K_{8,8}: vertices 0-7 in A, 8-15 in B
    # Triangle-free => every edge is diameter-critical when diameter=2
    # Mantel's theorem bounds triangle-free edges on 16 vertices at 64
    edges = []
    for u in range(8):
        for v in range(8, 16):
            edges.append([u, v])
    # Verify types are correct
    result = []
    for e in edges:
        result.append([int(e[0]), int(e[1])])
    return result
# EVOLVE-BLOCK-END
