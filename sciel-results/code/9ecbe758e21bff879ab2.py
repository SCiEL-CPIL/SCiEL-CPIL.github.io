# EVOLVE-BLOCK-START

def construct():
    """
    Returns edges for a 12-vertex graph (0..11) that is:
    - Simple, undirected
    - Triangle-free (girth >= 4)
    - Every vertex has degree >= 3
    - Contains cycles
    - Minimizes the number of distinct simple cycle lengths.

    The optimal raw value is 2, achieved by two disjoint K_{3,3} components.
    K_{3,3} is bipartite, so all cycles are even.
    In K_{3,3}, the shortest cycles are length 4 (C4) and length 6 (C6).
    There are no other simple cycle lengths in K_{3,3} (max cycle length is 6).
    Thus, the set of distinct cycle lengths is {4, 6}, count = 2.
    """
    edges = []
    
    # Component 1: K_{3,3} on vertices 0,1,2 (left) and 3,4,5 (right)
    left1 = (0, 1, 2)
    right1 = (3, 4, 5)
    for u in left1:
        for v in right1:
            edges.append([u, v])
            
    # Component 2: K_{3,3} on vertices 6,7,8 (left) and 9,10,11 (right)
    left2 = (6, 7, 8)
    right2 = (9, 10, 11)
    for u in left2:
        for v in right2:
            edges.append([u, v])
            
    return edges

# EVOLVE-BLOCK-END
