"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

P = {'n': 15}

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = 60.0 # Seconds
    
    # Construct K_{7,8}
    # Partition A: 0..6 (7 vertices)
    # Partition B: 7..14 (8 vertices)
    # Edges between all pairs in A and B
    edges = []
    for u in range(7):
        for v in range(7, 15):
            edges.append([u, v])
            
    # Check time, though this is instantaneous
    if time.time() - start_time > deadline:
        return edges
        
    return edges
# EVOLVE-BLOCK-END
