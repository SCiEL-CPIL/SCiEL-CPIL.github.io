def construct():
    """
    Constructs a graph on vertices 0..11 with minimum degree 3,
    triangle-free, and minimizing the number of distinct simple cycle lengths.
    
    The strategy is to use two disjoint K_{3,3} components.
    K_{3,3} is bipartite and 3-regular.
    The simple cycles in K_{3,3} have lengths 4 and 6.
    Thus, the set of distinct cycle lengths for the entire graph is {4, 6}.
    Raw value = 2.
    Score = 5.0 / 2.0 = 2.5.
    
    Any 3-regular bipartite graph on more than 6 vertices contains a 6-cycle.
    The smallest 3-regular bipartite graph is K_{3,3} (6 vertices).
    Since we have 12 vertices, we can form two K_{3,3} components.
    We cannot form a single component on 12 vertices with only one cycle length 
    because any 3-regular bipartite graph on 12 vertices has girth 4 (contains 4-cycles) 
    and necessarily contains 6-cycles.
    Non-bipartite options introduce odd cycles or higher girths that likely result 
    in more distinct cycle lengths.
    """
    edges = []
    
    # Component 1: K_{3,3} on vertices 0,1,2 (left) and 3,4,5 (right)
    for i in range(3):
        for j in range(3):
            u = int(i)
            v = int(3 + j)
            edges.append([u, v])
    
    # Component 2: K_{3,3} on vertices 6,7,8 (left) and 9,10,11 (right)
    for i in range(3):
        for j in range(3):
            u = int(6 + i)
            v = int(9 + j)
            edges.append([u, v])
    
    # Ensure edges are list of lists of Python ints
    result = [[int(u), int(v)] for u, v in edges]
    
    return result
