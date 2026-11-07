import time
from collections import deque

def construct():
    n = 16
    half = n // 2
    
    # K_{8,8} is the maximum triangle-free graph on 16 vertices (Turán's theorem).
    # For any edge (u,v) in a triangle-free graph, N(u) ∩ N(v) = ∅.
    # Thus removing edge (u,v) means u and v have no common neighbor,
    # so either they're disconnected or their distance becomes > 2.
    # In K_{8,8}, removing any edge (u,v) disconnects u and v entirely
    # (since u's neighbors are all in the other partition, and v is in that partition,
    #  but v is removed from u's adjacency... wait, no: u is still connected to all
    #  other vertices in v's partition, and v is connected to all other vertices in u's
    #  partition. But is there a path u -> w -> v? w would need to be adjacent to both u and v.
    #  u is in partition A, so its neighbors are in partition B. v is in partition B, so its
    #  neighbors are in partition A. A common neighbor w would need to be in both A and B,
    #  which is impossible. So no common neighbor exists, and after removing (u,v),
    #  u and v are disconnected.
    #
    # For non-adjacent pairs (both in same partition), they share all 8 vertices in the
    #  other partition as common neighbors, so diameter is 2.
    #
    # Therefore K_{8,8} satisfies all requirements and has the maximum possible edges
    #  for a triangle-free graph on 16 vertices.
    
    edges = []
    for u in range(half):
        for v in range(half, n):
            edges.append([int(u), int(v)])
    
    return edges
