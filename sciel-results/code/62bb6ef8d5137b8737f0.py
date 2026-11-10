import time

def construct():
    """
    Constructs a complete bipartite graph K_{7,8} on vertices 0..14.
    This graph is valid because any 3-path s-u-v-t has endpoints s and t
    in different partitions. The degrees of vertices in partition A (size 7)
    are 8, and in partition B (size 8) are 7. Since 8 != 7, the endpoints
    always have different degrees.
    """
    n = 15
    # Partition 1: 0 to 6 (7 vertices)
    # Partition 2: 7 to 14 (8 vertices)
    part1 = list(range(0, 7))
    part2 = list(range(7, 15))
    
    edges = []
    for u in part1:
        for v in part2:
            edges.append([u, v])
            
    return edges
