"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}

def construct():
    start = time.time()
    
    center = 12
    # Left side odds: 1, 3, 5, 7, 9
    # Right side odds: 15, 17, 19, 21, 23
    left_spokes = [1, 3, 5, 7, 9]
    right_spokes = [15, 17, 19, 21, 23]
    
    final_edges = []
    for node in left_spokes:
        final_edges.append([center, node])
    for node in right_spokes:
        final_edges.append([center, node])
        
    return [[int(u), int(v)] for u, v in final_edges]
# EVOLVE-BLOCK-END
