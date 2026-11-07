"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    start_time = time.time()
    deadline = start_time + 55  # Safety margin for 90s wall-clock limit

    n = 16
    half = n // 2

    # K_{8,8} is the optimal solution:
    # 1. Triangle-free (required: no edge can be in a triangle, else it's non-critical)
    # 2. Diameter 2 (any two vertices in same partition have all 8 vertices in other partition as common neighbors)
    # 3. Every edge is critical (removing edge (u,v) with u in A, v in B: no common neighbor exists
    #    since N(u) subset B and N(v) subset A, so intersection is empty; dist becomes 3)
    # 4. By Mantel's theorem, triangle-free graph on 16 vertices has at most floor(16^2/4) = 64 edges.
    #    K_{8,8} achieves exactly 64, so it is provably optimal.

    edges = []
    for u in range(half):
        for v in range(half, n):
            edges.append([u, v])

    # Verify we're within time budget (trivially true, but guard per directive)
    if time.time() > deadline:
        return edges

    return edges
# EVOLVE-BLOCK-END
