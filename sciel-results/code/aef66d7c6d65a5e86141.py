"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}

def construct():
    # The base graph is a path P_24.
    # We add edges from a central vertex (12) to all odd vertices, 
    # excluding those already connected by the path (11 and 13).
    # This keeps the graph bipartite (triangle-free) and reduces diameter to 4.
    
    center = 12
    added_edges = []
    
    # Odd vertices in 0..23
    odd_vertices = [v for v in range(P['n']) if v % 2 != 0]
    
    for v in odd_vertices:
        # Check if edge (center, v) is already in the path
        # Path edges are (i, i+1). 
        # Center is 12. Neighbors in path are 11 and 13.
        if abs(center - v) == 1:
            continue
        
        # Ensure u < v for consistent representation if needed, though task says [u,v]
        u, w = (center, v) if center < v else (v, center)
        added_edges.append([u, w])
        
    # The resulting graph has diameter 4 and is triangle-free.
    # Number of added edges is 10.
    return added_edges
# EVOLVE-BLOCK-END
