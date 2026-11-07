"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

P = {'n': 16}

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 59
    
    n = P['n']
    half = n // 2
    
    # Construct K_{half, half}
    # This is the unique triangle-free graph with maximum edges (Mantel's Theorem).
    # It has diameter 2.
    # For any edge (u, v), there is no other path of length 2 (triangle-free).
    # Thus, removing (u, v) makes dist(u, v) >= 3, increasing diameter.
    
    part_a = list(range(half))
    part_b = list(range(half, n))
    
    result = []
    for u in part_a:
        for v in part_b:
            result.append([u, v])
            
    return result
# EVOLVE-BLOCK-END
