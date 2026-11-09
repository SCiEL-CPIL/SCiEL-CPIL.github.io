"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin for 90s limit
    
    n = 16
    # K_{8,8} is the unique triangle-free graph on 16 vertices with maximum edges (64).
    # Triangle-free is required for every edge to be critical (removing an edge in a triangle
    # leaves a path of length 2, keeping diameter <= 2 if graph is dense enough).
    # K_{8,8} has diameter 2.
    
    edges = []
    for a in range(8):
        for b in range(8, 16):
            if time.time() > deadline:
                return edges
            edges.append([a, b])
            
    return edges
# EVOLVE-BLOCK-END
