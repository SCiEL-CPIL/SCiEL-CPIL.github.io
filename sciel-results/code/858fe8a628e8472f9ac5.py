"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START

def construct():
    # Construct two disjoint copies of K_{3,3}
    # K_{3,3} has vertices split into two sets of 3, all edges between sets.
    # Cycle lengths in K_{3,3} are 4 and 6.
    # Two disjoint copies result in the same set of cycle lengths {4, 6}.
    # This minimizes the number of distinct cycle lengths to 2.
    
    edges = []
    
    # First K_{3,3}: vertices 0,1,2 connected to 3,4,5
    # Part A: 0, 1, 2
    # Part B: 3, 4, 5
    for u in [0, 1, 2]:
        for v in [3, 4, 5]:
            edges.append([u, v])
            
    # Second K_{3,3}: vertices 6,7,8 connected to 9,10,11
    # Part A: 6, 7, 8
    # Part B: 9, 10, 11
    for u in [6, 7, 8]:
        for v in [9, 10, 11]:
            edges.append([u, v])
            
    # Ensure edges are lists of ints
    # No loops, no duplicates
    # Sort for consistency (optional but good practice)
    # Convert to Python int explicitly to satisfy type checks
    clean_edges = []
    for e in edges:
        u = int(e[0])
        v = int(e[1])
        if u != v:
            clean_edges.append([u, v])
            
    # Remove duplicates by sorting and unique-ing
    # Sort by min, then max
    clean_edges.sort(key=lambda x: (min(x), max(x)))
    
    final_edges = []
    prev = None
    for e in clean_edges:
        if e != prev:
            final_edges.append(e)
            prev = e
            
    return final_edges

# EVOLVE-BLOCK-END
