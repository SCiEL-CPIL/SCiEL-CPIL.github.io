import time

def construct():
    # The optimal solution for this problem on 12 vertices with min degree 3 
    # and triangle-free constraint appears to be two disjoint K_{3,3} components.
    # K_{3,3} has cycle lengths 4 and 6.
    # Two disjoint K_{3,3} components have cycle lengths {4, 6}.
    # Number of distinct cycle lengths = 2.
    # Score = seed / 2.
    
    edges = []
    
    # First K_{3,3} on vertices 0..5
    part_a1 = [0, 1, 2]
    part_b1 = [3, 4, 5]
    for u in part_a1:
        for v in part_b1:
            edges.append([u, v])
            
    # Second K_{3,3} on vertices 6..11
    part_a2 = [6, 7, 8]
    part_b2 = [9, 10, 11]
    for u in part_a2:
        for v in part_b2:
            edges.append([u, v])
            
    return edges
