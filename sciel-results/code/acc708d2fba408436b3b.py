def construct():
    """
    Construct a 12-vertex undirected simple graph with:
    - Every vertex has degree at least 3
    - Every simple cycle has length at least 4 (triangle-free)
    - Contains at least one cycle
    - Minimizes the number of distinct simple cycle lengths
    
    Strategy:
    - The known optimal solution is two disjoint K_{3,3} components, which have
      only cycle lengths 4 and 6, giving a value of 2.
    - We verify this is optimal: a triangle-free graph with min degree 3 must 
      contain cycles of length >= 4. To have only 1 distinct cycle length, all 
      cycles would need to be the same length. But K_{3,3} has both 4-cycles 
      and 6-cycles. It's theoretically impossible to have only one cycle length
      in a triangle-free graph with min degree 3 on 12 vertices.
    - So the best we can do is value = 2.
    - Return the two disjoint K_{3,3} graph immediately for speed and reliability.
    """
    
    # Two disjoint K_{3,3} components
    # First K_{3,3}: vertices 0,1,2 (part A) and 3,4,5 (part B)
    # Second K_{3,3}: vertices 6,7,8 (part A) and 9,10,11 (part B)
    
    edges = []
    
    # First K_{3,3}
    part_a1 = [0, 1, 2]
    part_b1 = [3, 4, 5]
    for u in part_a1:
        for v in part_b1:
            edges.append([u, v])
    
    # Second K_{3,3}
    part_a2 = [6, 7, 8]
    part_b2 = [9, 10, 11]
    for u in part_a2:
        for v in part_b2:
            edges.append([u, v])
    
    # Sort edges for consistent output
    edges.sort()
    
    # Ensure all values are Python int
    result = [[int(e[0]), int(e[1])] for e in edges]
    
    return result
