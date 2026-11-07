"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

P = {'n': 16}

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 59  # Safety margin within 60s limit
    
    n = P['n']
    half = n // 2
    
    # The optimal graph in the bipartite family is K_{n/2, n/2}.
    A = list(range(half))
    B = list(range(half, n))
    
    edges = []
    for u in A:
        for v in B:
            edges.append([u, v])
            
    return edges
# EVOLVE-BLOCK-END
