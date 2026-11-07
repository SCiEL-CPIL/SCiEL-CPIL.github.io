"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START

def construct():
    """
    Constructs a graph on 12 vertices (0..11) consisting of two disjoint copies of K_{3,3}.
    K_{3,3} is a complete bipartite graph with parts of size 3.
    - All vertices have degree 3.
    - It is triangle-free (bipartite).
    - The simple cycle lengths in K_{3,3} are 4 and 6.
    - Two disjoint copies do not create new cycles.
    - Distinct cycle lengths: {4, 6}. Count = 2.
    """
    edges = []
    
    # First K_{3,3}: vertices 0,1,2 (Part A) and 3,4,5 (Part B)
    part_a_1 = [0, 1, 2]
    part_b_1 = [3, 4, 5]
    for u in part_a_1:
        for v in part_b_1:
            edges.append([u, v])
            
    # Second K_{3,3}: vertices 6,7,8 (Part A) and 9,10,11 (Part B)
    part_a_2 = [6, 7, 8]
    part_b_2 = [9, 10, 11]
    for u in part_a_2:
        for v in part_b_2:
            edges.append([u, v])
            
    # Ensure edges are lists of Python ints
    # No loops, no duplicates
    clean_edges = []
    seen = set()
    for e in edges:
        u = int(e[0])
        v = int(e[1])
        if u == v:
            continue
        # Normalize order for duplicate checking
        if u > v:
            u, v = v, u
        if (u, v) not in seen:
            seen.add((u, v))
            clean_edges.append([u, v])
            
    return clean_edges

# EVOLVE-BLOCK-END
