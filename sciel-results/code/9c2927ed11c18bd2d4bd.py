import time

def construct():
    """
    Constructs a graph on 12 vertices (0..11) that minimizes the number of distinct
    simple cycle lengths.
    
    The optimal construction is two disjoint K_{3,3} components.
    Properties:
    - Simple, undirected, bipartite.
    - Minimum degree = 3.
    - Girth = 4.
    - Cycle lengths present: 4 and 6 only.
    - Distinct cycle lengths count: 2.
    
    This is optimal because:
    1. Value 1 is impossible for min-degree-3 triangle-free graphs on 12 vertices 
       (such graphs must contain cycles of at least two distinct lengths, typically 4 and 6).
    2. The K_{3,3} union achieves value 2.
    """
    
    # Construct edges for two disjoint K_{3,3} components
    edges = []
    
    # Component 1: K_{3,3} on vertices {0, 1, 2} and {3, 4, 5}
    left1 = [0, 1, 2]
    right1 = [3, 4, 5]
    for u in left1:
        for v in right1:
            edges.append([u, v])
            
    # Component 2: K_{3,3} on vertices {6, 7, 8} and {9, 10, 11}
    left2 = [6, 7, 8]
    right2 = [9, 10, 11]
    for u in left2:
        for v in right2:
            edges.append([u, v])
            
    # Ensure types are Python int and structure is list of lists
    result = []
    for e in edges:
        u, v = e
        result.append([int(u), int(v)])
        
    return result
