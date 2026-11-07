"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = 55.0 # Seconds, safety margin
    
    # Construct K_{7,8}
    # Partition A: 0..6 (7 vertices)
    # Partition B: 7..14 (8 vertices)
    # Edges between all pairs in A and B
    # This graph has 56 edges.
    # Degrees: A vertices have degree 8, B vertices have degree 7.
    # Any path of length 3 s-u-v-t has s and t in different partitions.
    # Thus deg(s) != deg(t) (8 != 7).
    # This satisfies the condition.
    
    edges = []
    for u in range(7):
        for v in range(7, 15):
            edges.append([u, v])
            
    # Verify types
    # Edges must be list of lists of ints
    # u, v are python ints from range()
    
    if time.time() - start_time > deadline:
        return edges
        
    return edges
# EVOLVE-BLOCK-END
