"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    # Two disjoint K_{3,3} components: vertices 0-5 and 6-11
    # Each K_{3,3} is 3-regular, bipartite, triangle-free, girth 4
    # Cycle lengths in K_{3,3}: 4 and 6 only
    # Distinct cycle lengths: {4, 6} -> raw value 2
    # Seed had raw value 5 (cube graph: 4,6,8,10,12), so score = 5/2 = 2.5
    edges = []
    for offset in (0, 6):
        left = [offset, offset + 1, offset + 2]
        right = [offset + 3, offset + 4, offset + 5]
        for u in left:
            for v in right:
                edges.append([int(u), int(v)])
    return edges
# EVOLVE-BLOCK-END
