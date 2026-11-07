"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START

import time

P = {'n': 16}
DEADLINE = 60.0

def construct():
    """
    Returns K_{8,8}, which is the optimal solution.
    
    Key insight: An edge (u,v) is critical (removing it increases diameter > 2 or disconnects)
    if and only if u and v have NO common neighbor.
    
    Therefore, we need a triangle-free graph with diameter exactly 2.
    By Turan's theorem, the maximum number of edges in a triangle-free graph on n vertices
    is floor(n^2/4), achieved by the complete bipartite graph K_{n/2, n/2}.
    
    For n=16: floor(16^2/4) = 64 edges, achieved by K_{8,8}.
    K_{8,8} has diameter 2. Every edge is critical.
    
    This is the theoretical maximum, so 64 edges is optimal.
    """
    start_time = time.time()
    n = P['n']
    half = n // 2
    
    edges = []
    for u in range(half):
        # Check deadline, though this loop is instant
        if time.time() - start_time > DEADLINE - 0.1:
            break
        for v in range(half, n):
            edges.append([int(u), int(v)])
    
    return edges
# EVOLVE-BLOCK-END
