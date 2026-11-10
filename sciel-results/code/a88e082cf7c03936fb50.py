"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START

import time

def construct():
    """
    Constructs a graph on vertices 0..11 consisting of two disjoint K_{3,3} components.
    
    K_{3,3} is a bipartite graph with partitions of size 3.
    It has minimum degree 3.
    It is triangle-free (girth 4).
    The simple cycles in K_{3,3} have lengths 4 and 6 only.
    Thus, the set of distinct cycle lengths is {4, 6}, and the count is 2.
    
    This achieves a raw value of 2.
    """
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin before 90s kill
    
    edges = []
    
    # Component 1: Vertices 0,1,2 (Part A) and 3,4,5 (Part B)
    part_a_1 = [0, 1, 2]
    part_b_1 = [3, 4, 5]
    for u in part_a_1:
        for v in part_b_1:
            edges.append([u, v])
            
    # Component 2: Vertices 6,7,8 (Part A) and 9,10,11 (Part B)
    part_a_2 = [6, 7, 8]
    part_b_2 = [9, 10, 11]
    for u in part_a_2:
        for v in part_b_2:
            edges.append([u, v])
            
    # Verify: check all vertices have degree >= 3
    degree = [0] * 12
    for u, v in edges:
        degree[u] += 1
        degree[v] += 1
    assert all(d >= 3 for d in degree), f"Min degree violated: {degree}"
    
    # Verify: no triangles (girth >= 4)
    adj = set()
    for u, v in edges:
        adj.add((min(u,v), max(u,v)))
    for i in range(12):
        for j in range(i+1, 12):
            if (i, j) in adj:
                for k in range(j+1, 12):
                    if (i, k) in adj and (j, k) in adj:
                        raise ValueError(f"Triangle found: {i},{j},{k}")
    
    # Verify: at least one cycle exists
    # K_{3,3} has cycles, so this is guaranteed
    
    # Ensure proper types: int, list
    result = []
    for edge in edges:
        result.append([int(edge[0]), int(edge[1])])
    
    return result

# EVOLVE-BLOCK-END
