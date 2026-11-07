"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START

def construct():
    # Build two disjoint K3,3 components on vertices 0..11
    # Component 1: vertices 0,1,2 (part A) and 3,4,5 (part B)
    # Component 2: vertices 6,7,8 (part A) and 9,10,11 (part B)
    #
    # Each K3,3 has minimum degree 3, is bipartite (hence triangle-free),
    # and its simple cycles have lengths 4 and 6 only.
    # Distinct cycle lengths: {4, 6} -> count = 2

    result = []
    
    # First K3,3
    for i in range(3):
        for j in range(3):
            u = i
            v = 3 + j
            result.append([u, v])
    
    # Second K3,3
    for i in range(3):
        for j in range(3):
            u = 6 + i
            v = 9 + j
            result.append([u, v])
    
    return result

# EVOLVE-BLOCK-END
