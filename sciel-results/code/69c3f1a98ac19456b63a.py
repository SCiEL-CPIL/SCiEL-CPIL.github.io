import time

def construct():
    # The two disjoint K_{3,3} graph is the optimal solution with 2 distinct cycle lengths (4 and 6).
    # It satisfies all constraints:
    # - 12 vertices (0..11)
    # - Every vertex has degree 3 (>= 3)
    # - Triangle-free (bipartite)
    # - Contains cycles
    # - Number of distinct cycle lengths is 2 (minimum possible)
    
    edges = []
    # Component 1: Vertices 0,1,2 (L) and 3,4,5 (R)
    for i in range(3):
        for j in range(3):
            u = i
            v = j + 3
            edges.append([u, v])
    
    # Component 2: Vertices 6,7,8 (L) and 9,10,11 (R)
    for i in range(3):
        for j in range(3):
            u = i + 6
            v = j + 9
            edges.append([u, v])
    
    return edges
