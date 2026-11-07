"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    # Construct two disjoint K_{3,3} components.
    # K_{3,3} is bipartite, triangle-free, min degree 3.
    # Cycle lengths in K_{3,3} are 4 and 6.
    # Distinct cycle lengths: {4, 6} -> count 2.
    # This achieves the minimum possible distinct cycle length count for this problem size
    # under the constraints, yielding a score of 5/2 = 2.5.
    
    edges = []
    
    # Component 1: K_{3,3} on vertices 0..5
    # Part U = {0, 1, 2}, Part V = {3, 4, 5}
    for u in (0, 1, 2):
        for v in (3, 4, 5):
            edges.append([u, v])
            
    # Component 2: K_{3,3} on vertices 6..11
    # Part U = {6, 7, 8}, Part V = {9, 10, 11}
    for u in (6, 7, 8):
        for v in (9, 10, 11):
            edges.append([u, v])
            
    # Ensure types are Python int and format is list of lists
    result = []
    for e in edges:
        result.append([int(e[0]), int(e[1])])
        
    return result
# EVOLVE-BLOCK-END
