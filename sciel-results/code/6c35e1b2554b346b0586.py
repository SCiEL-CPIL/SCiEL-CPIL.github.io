import time

N = 12

def construct():
    """
    Constructs a graph on 12 vertices with minimum degree 3, triangle-free,
    and minimizing the number of distinct simple cycle lengths.
    
    The optimal solution found is two disjoint K_{3,3} components.
    K_{3,3} has cycle lengths 4 and 6.
    Thus, the set of distinct cycle lengths is {4, 6}, count = 2.
    
    It is hypothesized that 2 is the minimum possible count for 12 vertices
    with min degree 3 and girth >= 4.
    """
    edges = []
    
    # Component 1: K_{3,3} on vertices 0, 1, 2 and 3, 4, 5
    part_a1 = [0, 1, 2]
    part_b1 = [3, 4, 5]
    for u in part_a1:
        for v in part_b1:
            edges.append([u, v])
            
    # Component 2: K_{3,3} on vertices 6, 7, 8 and 9, 10, 11
    part_a2 = [6, 7, 8]
    part_b2 = [9, 10, 11]
    for u in part_a2:
        for v in part_b2:
            edges.append([u, v])
            
    return edges
