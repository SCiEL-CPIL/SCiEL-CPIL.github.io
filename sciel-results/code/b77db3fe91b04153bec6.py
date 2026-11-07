"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START

def construct():
    # Construct a connected graph consisting of two K_{3,3} components
    # joined by a single bridge edge.
    # This ensures the graph is connected (unlike the seed which was disconnected).
    # The cycles are still only those from the K_{3,3} components, as the bridge
    # cannot be part of any simple cycle.
    # Thus, the set of distinct cycle lengths remains {4, 6}, with raw value 2.
    
    edges = []
    
    # Component 1: K_{3,3} on vertices 0..5
    # Partition: A = {0, 1, 2}, B = {3, 4, 5}
    part1_a = [0, 1, 2]
    part1_b = [3, 4, 5]
    for u in part1_a:
        for v in part1_b:
            edges.append([u, v])
    
    # Component 2: K_{3,3} on vertices 6..11
    # Partition: A = {6, 7, 8}, B = {9, 10, 11}
    part2_a = [6, 7, 8]
    part2_b = [9, 10, 11]
    for u in part2_a:
        for v in part2_b:
            edges.append([u, v])
            
    # Bridge edge connecting the two components
    # Connect vertex 0 (from comp 1) to vertex 6 (from comp 2)
    edges.append([0, 6])
    
    # Verification:
    # - Simple: No loops, no duplicates.
    # - Degree: Vertices 0 and 6 have degree 4. Others have degree 3. Min degree 3.
    # - Bipartite: K_{3,3} is bipartite. The bridge connects 0 (in A1) to 6 (in A2).
    #   Wait, 0 is in A1. 6 is in A2.
    #   In K_{3,3}, A and B are the partitions.
    #   If we connect A1 to A2, is the whole graph bipartite?
    #   Let's check the parity.
    #   Path from 0 to 6 in the combined graph?
    #   0 is in A1. 6 is in A2.
    #   In K_{3,3}, any path from A to A has even length.
    #   If we add edge (0,6), and 0,6 are in the "same" partition relative to their components,
    #   we might create an odd cycle if there was a path between them.
    #   But the components were disconnected.
    #   Now they are connected.
    #   Is the resulting graph bipartite?
    #   Let's assign colors.
    #   Comp 1: 0,1,2 are Black; 3,4,5 are White.
    #   Comp 2: 6,7,8 are Black; 9,10,11 are White.
    #   Edge (0,6) connects Black to Black.
    #   This creates an odd cycle?
    #   No, it just means the coloring is not consistent if we assume both are "Black".
    #   We can recolor Comp 2.
    #   If we flip Comp 2 colors: 6,7,8 become White; 9,10,11 become Black.
    #   Then edge (0,6) connects Black (0) to White (6).
    #   This is a valid bipartite coloring.
    #   So the graph is bipartite.
    # - Triangle-free: Bipartite graphs have no odd cycles, so no triangles.
    # - Cycles: Only 4 and 6 from K_{3,3}s.
    
    return edges

# EVOLVE-BLOCK-END
