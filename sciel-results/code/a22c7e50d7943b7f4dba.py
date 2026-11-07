"""Rewritten program following the research directive."""
# EVOLVE-BLOCK-START
def construct():
    edges = []
    # Construct two disjoint K_{3,3} graphs
    # First K_{3,3} on vertices 0-5
    left1 = [0, 1, 2]
    right1 = [3, 4, 5]
    for u in left1:
        for v in right1:
            edges.append([u, v])
            
    # Second K_{3,3} on vertices 6-11
    left2 = [6, 7, 8]
    right2 = [9, 10, 11]
    for u in left2:
        for v in right2:
            edges.append([u, v])
            
    return edges
# EVOLVE-BLOCK-END
