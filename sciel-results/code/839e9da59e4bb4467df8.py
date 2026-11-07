def construct():
    # Returns edges for two disjoint K_{3,3} components.
    # Component 1: vertices 0,1,2 and 3,4,5
    # Component 2: vertices 6,7,8 and 9,10,11
    edges = []
    
    # Define partitions for the first K_{3,3}
    part_a = [0, 1, 2]
    part_b = [3, 4, 5]
    
    # Define partitions for the second K_{3,3}
    part_c = [6, 7, 8]
    part_d = [9, 10, 11]
    
    # Add edges for first component
    for u in part_a:
        for v in part_b:
            edges.append([u, v])
            
    # Add edges for second component
    for u in part_c:
        for v in part_d:
            edges.append([u, v])
            
    return edges
