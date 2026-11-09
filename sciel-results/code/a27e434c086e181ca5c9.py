"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    # Deadline guard: return early if we are running out of time, 
    # though this construction is O(N^2) and very fast.
    start_time = time.time()
    deadline = 85.0  # 90s limit, keep 5s margin
    
    n = 16
    # Split vertices into two sets of size 8
    # Set A: 0..7, Set B: 8..15
    edges = []
    for u in range(8):
        for v in range(8, 16):
            edges.append([u, v])
            
    # The graph is K_{8,8}, which has 64 edges.
    # It is triangle-free, so every edge is critical (removing it increases distance to 3).
    # It has diameter 2.
    
    if time.time() > deadline:
        # Fallback to a smaller valid graph if somehow we are slow (unlikely)
        # But K_{8,8} is instant.
        pass
        
    return edges

# EVOLVE-BLOCK-END
