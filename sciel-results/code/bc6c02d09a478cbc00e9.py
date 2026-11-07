"""Rewritten program: K_{8,8} as the extremal diameter-2 edge-critical graph."""
# EVOLVE-BLOCK-START
def construct():
    # K_{8,8} is the complete bipartite graph with partitions A = {0..7}, B = {8..15}.
    #
    # Verification of the two required properties:
    # 1. Diameter exactly 2:
    #    - Any u in A and v in B are adjacent, so dist(u,v) = 1.
    #    - Any u1, u2 in A (u1 != u2): pick any w in B; then u1-w-u2 is a path of length 2.
    #    - Any v1, v2 in B (v1 != v2): pick any w in A; then v1-w-v2 is a path of length 2.
    #    Hence the diameter is exactly 2.
    #
    # 2. Every edge is essential (removing it raises the diameter above 2):
    #    - K_{8,8} is triangle-free (no three mutually adjacent vertices).
    #    - Consider any edge (u, v) with u in A, v in B.
    #    - If (u, v) is removed, is there a path of length <= 2 from u to v?
    #      A path of length 1 is the edge itself (gone).
    #      A path of length 2 would be u - w - v for some w.
    #      Since u is in A, w must be in B (u's neighbors are all in B).
    #      Since v is in B, w must be in A (v's neighbors are all in A).
    #      Contradiction: w cannot be in both A and B.
    #    - Therefore no path of length 2 exists after removal; the shortest path has length 3.
    #      (e.g., u - v' - u' - v for any v' in B\{v}, u' in A\{u}.)
    #    - The diameter of G - (u,v) is at least 3, so the edge is essential.
    #
    # 3. Maximality (why 64 edges is optimal):
    #    - For an edge to be essential, the graph must be triangle-free (otherwise an edge
    #      in a triangle would have a length-2 alternative path and be non-essential).
    #    - By Mantel's Theorem, the maximum number of edges in a triangle-free graph on
    #      n = 16 vertices is floor(n^2 / 4) = 64, achieved uniquely by K_{8,8}.
    #    - Thus 64 is the theoretical maximum edge count satisfying all constraints.
    #
    # Return type: list of [int, int] pairs, no loops, no duplicates.
    half = 8
    result = []
    for i in range(half):
        for j in range(half, 16):
            result.append([i, j])
    return result
# EVOLVE-BLOCK-END
