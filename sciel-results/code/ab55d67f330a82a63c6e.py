"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    
    # Construct 8 disjoint copies of K4.
    # Total vertices n = 8 * 4 = 32.
    # No C5 (components are K4).
    # Independence number alpha = 8 (one per K4).
    # This satisfies R(C5, K9) > 32.
    
    n = 32
    edges = []
    
    for b in range(8):
        base = 4 * b
        for i in range(4):
            for j in range(i + 1, 4):
                edges.append([base + i, base + j])
                
    return {'n': n, 'edges': edges}
# EVOLVE-BLOCK-END
