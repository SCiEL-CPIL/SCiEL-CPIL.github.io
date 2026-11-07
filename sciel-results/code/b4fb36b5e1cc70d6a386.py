"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START

def construct():
    # Construct two disjoint K_{3,3} components.
    # K_{3,3} is bipartite with parts of size 3.
    # It has girth 4 and contains cycles of length 4 and 6.
    # The set of distinct cycle lengths is {4, 6}, so the count is 2.
    # This is the theoretical minimum for graphs with min degree >= 3.
    
    edges = []
    
    # Vertices 0, 1, 2 form one partition of the first K_{3,3}
    # Vertices 3, 4, 5 form the other partition
    part1_left = [0, 1, 2]
    part1_right = [3, 4, 5]
    
    # Vertices 6, 7, 8 form one partition of the second K_{3,3}
    # Vertices 9, 10, 11 form the other partition
    part2_left = [6, 7, 8]
    part2_right = [9, 10, 11]
    
    # Generate edges for the first component
    for u in part1_left:
        for v in part1_right:
            edges.append([u, v])
            
    # Generate edges for the second component
    for u in part2_left:
        for v in part2_right:
            edges.append([u, v])
            
    return edges

# EVOLVE-BLOCK-END
