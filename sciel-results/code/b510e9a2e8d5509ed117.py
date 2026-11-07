"""Dense diameter-2 graph on 16 vertices where every edge is critical."""
# EVOLVE-BLOCK-START

def construct():
    # Vertices are 0..15.  The optimal structure for this task is the
    # complete bipartite graph K_{8,8}: it is triangle-free (so no edge
    # has a 2-path alternative that would make it non-essential), has
    # diameter exactly 2, and removing any single edge forces its two
    # endpoints to distance 3 (via the other partition).
    #
    # We build the edge list with a single deterministic construction
    # that yields exactly 8*8 = 64 edges.
    left = [0, 1, 2, 3, 4, 5, 6, 7]
    right = [8, 9, 10, 11, 12, 13, 14, 15]

    edges = []
    for a in left:
        for b in right:
            edges.append([a, b])

    return edges

# EVOLVE-BLOCK-END
