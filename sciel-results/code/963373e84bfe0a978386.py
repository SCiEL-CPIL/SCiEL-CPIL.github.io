"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START

def construct():
    """Build two disjoint K_{3,3} components on vertices 0..11.
    
    Component 1: parts {0,1,2} and {3,4,5}
    Component 2: parts {6,7,8} and {9,10,11}
    
    Every vertex has degree exactly 3.
    The graph is bipartite (hence triangle-free, girth >= 4).
    K_{3,3} contains 4-cycles and 6-cycles only.
    Distinct cycle lengths = {4, 6} -> 2 total.
    """
    edges = []
    for offset in (0, 6):
        part_a = [offset, offset + 1, offset + 2]
        part_b = [offset + 3, offset + 4, offset + 5]
        for u in part_a:
            for v in part_b:
                edges.append([u, v])
    return edges

# EVOLVE-BLOCK-END
