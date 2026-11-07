"""Program that constructs a 12-vertex graph minimizing distinct cycle lengths."""
# EVOLVE-BLOCK-START

import time

def construct():
    deadline = time.time() + 55.0
    
    # Two disjoint K_{3,3} components:
    # - 12 vertices total (6+6)
    # - Minimum degree 3 (each vertex in K_{3,3} has degree 3)
    # - Bipartite, so girth 4 (no odd cycles, no triangles)
    # - Cycle lengths present: 4 and 6
    # - Distinct cycle lengths: {4, 6}, count = 2
    #
    # This is optimal: any graph with min degree >= 3 and girth >= 4 on 12 vertices
    # must have at least 2 distinct cycle lengths.
    
    edges = []
    
    # Component 1: K_{3,3} on vertices 0,1,2 (left) and 3,4,5 (right)
    for u in [0, 1, 2]:
        for v in [3, 4, 5]:
            edges.append([u, v])
    
    # Component 2: K_{3,3} on vertices 6,7,8 (left) and 9,10,11 (right)
    for u in [6, 7, 8]:
        for v in [9, 10, 11]:
            edges.append([u, v])
    
    return edges

# EVOLVE-BLOCK-END
