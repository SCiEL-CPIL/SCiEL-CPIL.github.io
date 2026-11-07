import time

def construct():
    start_time = time.time()
    deadline = 85.0

    # The theoretical minimum number of distinct cycle lengths for a graph
    # with minimum degree >= 3 and girth >= 4 is 2.
    # This is achieved by the complete bipartite graph K_{3,3}, which has
    # cycles of length 4 and 6 only.
    # We construct two disjoint copies of K_{3,3} to utilize all 12 vertices
    # while maintaining the minimum degree of 3 for all vertices.
    
    edges = []
    
    # Component 1: K_{3,3} on vertices 0..5
    # Part A: 0, 1, 2
    # Part B: 3, 4, 5
    part_a1 = [0, 1, 2]
    part_b1 = [3, 4, 5]
    for u in part_a1:
        for v in part_b1:
            edges.append([u, v])
            
    # Component 2: K_{3,3} on vertices 6..11
    # Part A: 6, 7, 8
    # Part B: 9, 10, 11
    part_a2 = [6, 7, 8]
    part_b2 = [9, 10, 11]
    for u in part_a2:
        for v in part_b2:
            edges.append([u, v])
            
    # Ensure types are correct
    final_edges = []
    for e in edges:
        final_edges.append([int(e[0]), int(e[1])])
        
    # Time guard
    if time.time() - start_time > deadline:
        pass
        
    return final_edges
