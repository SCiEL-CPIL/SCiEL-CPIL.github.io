"""Program for the suite task; constructs dense odd-order graphs avoiding equal-degree path endpoints."""

def construct():
    n = 15
    # Use complete bipartite graph K_{7,8}
    # Partition A: vertices 0..6 (size 7), each vertex has degree 8
    # Partition B: vertices 7..14 (size 8), each vertex has degree 7
    # In K_{7,8}, every 3-edge path s-u-v-t has s and t in opposite partitions
    # so deg(s) != deg(t) since 7 != 8
    a_size = 7
    b_size = 8
    partition_a = list(range(0, a_size))
    partition_b = list(range(a_size, n))
    edges = []
    for u in partition_a:
        for v in partition_b:
            edges.append([u, v])
    return edges
