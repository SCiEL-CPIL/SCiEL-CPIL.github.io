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
    
    It is highly probable that raw value 1 is unattainable for a 
    triangle-free graph with minimum degree 3 on 12 vertices, 
    as connected min-degree-3 graphs typically contain cycles 
    of multiple lengths, and disconnected components must 
    individually satisfy the degree constraint, forcing them 
    to be at least as complex as K_{3,3}.
    """
    import time
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s limit
    
    edges = []
    
    # Component 1: K_{3,3} on vertices 0,1,2 (left) and 3,4,5 (right)
    for i in range(3):
        for j in range(3):
            u = int(i)
            v = int(3 + j)
            edges.append([u, v])
            
    # Check deadline (should be immediate, but guard just in case)
    if time.time() > deadline:
        # Return partial result if we were doing complex search, 
        # but here it's static.
        pass
        
    # Component 2: K_{3,3} on vertices 6,7,8 (left) and 9,10,11 (right)
    for i in range(3):
        for j in range(3):
            u = int(6 + i)
            v = int(9 + j)
            edges.append([u, v])
            
    # Ensure edges are list of lists of Python ints
    result = [[int(u), int(v)] for u, v in edges]
    
    return result
