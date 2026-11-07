"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

# Deadline guard: keep a hard wall-clock limit with safety margin
_DEADLINE = time.time() + 85.0  # 5s margin before 90s kill

def construct():
    # K_{8,8} is the extremal graph:
    # - diameter exactly 2 (any two vertices in same part have common neighbor in other part)
    # - every edge is essential (removing any edge (u,v) makes dist(u,v) = 3, so diameter >= 3)
    # - triangle-free (necessary: if there were a triangle u-w-v, removing (u,v) would leave a
    #   path u-w-v of length 2, so that edge would NOT be essential)
    # - 64 edges is maximum possible for a diameter-2 graph on 16 vertices where every edge is essential
    n = 16
    a = n // 2  # 8
    edges = []
    for u in range(a):
        for v in range(a, n):
            edges.append([u, v])
    # Ensure all types are Python int and list
    edges = [[int(u), int(v)] for u, v in edges]
    return edges
# EVOLVE-BLOCK-END
