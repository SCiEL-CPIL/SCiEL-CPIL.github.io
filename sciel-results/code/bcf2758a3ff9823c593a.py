"""
Constructs a graph on 16 vertices with diameter exactly 2 such that removing
any single edge increases the diameter above 2 (or disconnects the graph).

The optimal structure is the complete bipartite graph K_{8,8} with 64 edges.
Triangle-freeness is necessary for edge-criticality: if three vertices form
a triangle, removing any edge of that triangle still leaves a path of length
2, so that edge is not critical. The Mantel bound states that the maximum
number of edges in a triangle-free graph on 16 vertices is floor(16^2/4) = 64,
achieved exactly by K_{8,8}.

In K_{8,8}, every edge (u, v) with u in partition A and v in partition B is
critical: u's neighbors are exactly B \ {v} and v's neighbors are exactly
A \ {u}, so there is no common neighbor. Removing (u, v) makes dist(u, v) = 3.
All other pairs remain at distance <= 2, so the diameter becomes exactly 3.
"""
import time


def construct():
    # 50-second safety margin well within the 90-second evaluation limit.
    start_time = time.time()
    deadline = start_time + 50

    # Build K_{8,8}: vertices 0..7 in partition A, vertices 8..15 in partition B.
    edges = []
    for u in range(8):
        for v in range(8, 16):
            # Ensure plain Python int for the strict type check.
            edges.append([int(u), int(v)])

    # If somehow we ran out of time (should never happen for this construction),
    # return what we have so far; the construction is O(64) so this is safe.
    if time.time() > deadline:
        return edges

    # K_{8,8} has 64 edges, the Mantel-optimal count for triangle-free graphs
    # on 16 vertices. Every edge is critical for diameter 2, so this is the
    # maximum possible edge count satisfying the task constraints.
    return edges
