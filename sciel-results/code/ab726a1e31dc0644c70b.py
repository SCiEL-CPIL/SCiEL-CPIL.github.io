"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}

def construct():
    # The optimal solution found involves connecting vertex 12 to all other odd vertices
    # except those already connected by the path (11 and 13).
    # This ensures the graph remains bipartite (triangle-free) and reduces the diameter to 4.
    
    # Odd vertices in 0..23
    odds = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23]
    
    # Vertices already connected to 12 in the path
    existing_neighbors = {11, 13}
    
    added_edges = []
    for o in odds:
        if o not in existing_neighbors:
            added_edges.append([12, o])
            
    return added_edges
# EVOLVE-BLOCK-END
