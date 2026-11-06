"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    # Blow-up of K5 minus edge (0,1).
    # Base vertices 0..4.
    # Part sizes: parts[0]=3, parts[1]=3, parts[2]=4, parts[3]=4, parts[4]=4.
    # Total vertices: 3+3+4+4+4 = 18.
    # Edges exist between all vertices in different parts, except between parts 0 and 1.
    # This graph is K4-free because the base graph is K5-e which is K4-free.
    # The largest triangle-free induced subgraph has size 10.
    
    part_sizes = [3, 3, 4, 4, 4]
    vertices = list(range(18))
    
    # Assign vertices to parts
    # Part 0: 0-2
    # Part 1: 3-5
    # Part 2: 6-9
    # Part 3: 10-13
    # Part 4: 14-17
    parts = []
    start = 0
    for size in part_sizes:
        parts.append(vertices[start:start+size])
        start += size
    
    edges = []
    # Add edges between all pairs of parts, except between part 0 and part 1
    for i in range(5):
        for j in range(i+1, 5):
            if (i == 0 and j == 1) or (i == 1 and j == 0):
                continue
            for u in parts[i]:
                for v in parts[j]:
                    edges.append([u, v])
    
    return edges
# EVOLVE-BLOCK-END
