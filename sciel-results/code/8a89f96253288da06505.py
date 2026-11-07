def construct():
    # Complete bipartite graph K_{7,8} on vertices 0..14.
    # A = {0,1,...,6}, B = {7,8,...,14}
    # All vertices in A have degree 8, all in B have degree 7.
    # Any 3-edge simple path s-u-v-t alternates partitions,
    # so s and t are in different partitions, hence degrees differ (8 vs 7).
    # This is the maximum possible edge count under the constraint.
    edges = []
    for i in range(7):
        for j in range(7, 15):
            edges.append([i, j])
    return edges
