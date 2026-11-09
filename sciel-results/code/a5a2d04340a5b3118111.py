"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    # Deadline guard (not strictly necessary for this O(n^2) task, but required by directive)
    start_time = time.time()
    deadline = start_time + 50.0  # 50s margin within 60s limit
    
    n = 16
    half = n // 2
    edges = []
    
    # Construct K_{8,8}
    # Vertices 0-7 in set A, 8-15 in set B
    # All edges between A and B
    for u in range(half):
        if time.time() > deadline:
            break
        for v in range(half, n):
            if time.time() > deadline:
                break
            edges.append([u, v])
            
    return edges
# EVOLVE-BLOCK-END
