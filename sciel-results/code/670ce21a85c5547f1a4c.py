import time

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 5.0
    
    n = 16
    # The optimal graph is K_{8,8}, a complete bipartite graph.
    # It has 64 edges, is triangle-free, and has diameter 2.
    # Every edge is critical because removing any edge (u,v) from a triangle-free 
    # diameter-2 graph forces the distance between u and v to become at least 3.
    
    # Partition vertices into two sets of 8
    # Set A: 0-7, Set B: 8-15
    set_a = range(0, 8)
    set_b = range(8, 16)
    
    edges = []
    
    # Generate all edges between set A and set B
    # Using a nested loop with explicit time checks to satisfy deadline guard requirements
    for i in set_a:
        if time.time() > deadline:
            # Fallback to a minimal valid graph if time runs out (unlikely for 64 edges)
            return [[0, 1], [1, 2], [0, 2]] 
        for j in set_b:
            edges.append([i, j])
            
    # Ensure types are strictly Python int and list
    # The loop above produces lists of ints, which is correct.
    return edges
