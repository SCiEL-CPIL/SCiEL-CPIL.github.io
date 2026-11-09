"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}

def construct():
    # The optimal solution for this specific instance (Path 24, Diameter 4, Triangle-Free)
    # uses a central hub (vertex 12) connected to all odd vertices except its path neighbors.
    # This ensures the graph remains bipartite (triangle-free) and achieves diameter 4.
    
    # Hub 12 is even.
    # Odd vertices are 1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23.
    # 11 and 13 are already connected to 12 via the path.
    # We must connect 12 to the other 10 odd vertices to ensure radius 2.
    
    hub = 12
    odd_vertices = [1, 3, 5, 7, 9, 15, 17, 19, 21, 23]
    
    added_edges = []
    for v in odd_vertices:
        added_edges.append([hub, v])
        
    return added_edges
# EVOLVE-BLOCK-END
