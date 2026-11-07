"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

P = {'n': 12, 'minimum_degree': 3, 'girth': 4}
START_TIME = time.time()
TIME_LIMIT = 55.0  # Safety margin for 60s limit

def construct():
    n = P['n']
    edges = []
    
    # Check time limit immediately
    if time.time() - START_TIME > TIME_LIMIT:
        return edges

    # Construct two disjoint K_{3,3} components.
    # This structure ensures:
    # 1. Minimum degree 3 (all vertices in K_{3,3} have degree 3).
    # 2. Girth 4 (K_{3,3} is bipartite, so no triangles. Smallest cycle is 4).
    # 3. Cycle lengths are 4 and 6. Distinct lengths count = 2.
    # This is the known minimum for this class of graphs.
    
    # Component 1: Vertices 0, 1, 2 connected to 3, 4, 5.
    for u in range(3):
        for v in range(3, 6):
            if time.time() - START_TIME > TIME_LIMIT:
                break
            edges.append([u, v])
        if time.time() - START_TIME > TIME_LIMIT:
            break
            
    if time.time() - START_TIME > TIME_LIMIT:
        return edges

    # Component 2: Vertices 6, 7, 8 connected to 9, 10, 11.
    for u in range(6, 9):
        for v in range(9, 12):
            if time.time() - START_TIME > TIME_LIMIT:
                break
            edges.append([u, v])
        if time.time() - START_TIME > TIME_LIMIT:
            break

    # Ensure types are Python native
    final_edges = []
    for e in edges:
        final_edges.append([int(e[0]), int(e[1])])
        
    return final_edges

# EVOLVE-BLOCK-END
