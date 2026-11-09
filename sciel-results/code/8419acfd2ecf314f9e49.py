"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    """
    Construct a 12-vertex graph with minimum degree >= 3, girth >= 4,
    minimizing the number of distinct simple cycle lengths.
    
    Strategy: Use two disjoint K_{3,3} components.
    Each K_{3,3} is 3-regular, bipartite, triangle-free, with girth 4.
    The only cycle lengths in K_{3,3} are 4 and 6.
    Thus distinct cycle lengths = {4, 6}, raw value = 2.
    
    This is optimal: any graph with min degree >= 3 must have cycles,
    and the smallest possible girth is 4 (triangle-free). A graph with
    only one cycle length would need to be a single girth-g cycle, but
    min degree 3 forces additional cycles of different lengths.
    """
    edges = []
    for offset in (0, 6):
        left = [offset, offset + 1, offset + 2]
        right = [offset + 3, offset + 4, offset + 5]
        for u in left:
            for v in right:
                edges.append([int(u), int(v)])
    return edges
# EVOLVE-BLOCK-END
