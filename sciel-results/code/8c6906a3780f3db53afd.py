"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START

def construct():
    edges = []
    
    # First K_{3,3} component on vertices 0..5
    # Part A: 0, 1, 2
    # Part B: 3, 4, 5
    part_a_1 = [0, 1, 2]
    part_b_1 = [3, 4, 5]
    for u in part_a_1:
        for v in part_b_1:
            edges.append([u, v])
            
    # Second K_{3,3} component on vertices 6..11
    # Part A: 6, 7, 8
    # Part B: 9, 10, 11
    part_a_2 = [6, 7, 8]
    part_b_2 = [9, 10, 11]
    for u in part_a_2:
        for v in part_b_2:
            edges.append([u, v])
            
    return edges

# EVOLVE-BLOCK-END
