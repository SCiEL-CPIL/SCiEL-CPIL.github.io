"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    start = time.time()
    deadline = start + 58.0

    n = 16
    half = n >> 1

    # Strategy: The complete bipartite graph K_{8,8} is the unique (up to iso)
    # triangle-free graph with the maximum number of edges (Mantel's Theorem: floor(n^2/4) = 64).
    #
    # Why this is optimal for "edge-critical diameter 2":
    # 1. Any graph with > 64 edges must contain a triangle.
    # 2. In a triangle (u, v, w), the edge (u, v) is not critical for diameter 2.
    #    If we remove (u, v), the path u-w-v of length 2 still exists, so dist(u, v) remains 2.
    #    Since the graph is connected and has diameter 2, removing an edge that doesn't increase
    #    any pair's distance beyond 2 (or disconnect) is not "critical" in the sense required
    #    (the task says "Deleting EACH individual edge must increase its diameter above two or disconnect it").
    #    Wait, let's re-read carefully: "Deleting EACH individual edge must increase its diameter above two or disconnect it."
    #    This implies that for EVERY edge e, G-e has diameter > 2 OR is disconnected.
    #    If a graph has a triangle, removing one edge of the triangle leaves a path of length 2 between those vertices.
    #    Does the diameter increase? Not necessarily for that pair. But could it for others?
    #    Actually, if a graph is edge-critical for diameter 2, it must be triangle-free.
    #    Proof: Suppose G has a triangle u-v-w-u. Remove edge (u, v).
    #    dist(u, v) in G-(u,v) is at most 2 (via w).
    #    For any other pair (x, y), if dist_G(x, y) <= 2, does dist_{G-(u,v)}(x, y) <= 2?
    #    If a shortest path in G uses (u, v), it can be replaced by (u, w, v). Length increases by 1.
    #    So a path of length 2 becomes length 3. This could increase the diameter.
    #    However, if the graph is K_{n,n}, it is triangle-free.
    #    Let's check if K_{8,8} is edge-critical.
    #    Remove edge (u, v) where u in A, v in B.
    #    dist(u, v) in G-e: u is connected to B\{v}, v is connected to A\{u}.
    #    Is there a path of length 2? u-w-v? w must be neighbor of u and v.
    #    Neighbors of u are in B. Neighbors of v are in A.
    #    Intersection of N(u) and N(v) is empty.
    #    So dist(u, v) >= 3.
    #    Path u-w1-v1-v? No, w1 in B, v1 in A. u-w1 (len 1), w1-v1 (len 1), v1-v (len 1). Total 3.
    #    So dist(u, v) = 3. Diameter becomes 3.
    #    So K_{8,8} IS edge-critical.
    #
    # Can we have a graph with > 64 edges that is edge-critical?
    # No, because it would have a triangle.
    # Let G be a graph with a triangle. Remove an edge (u, v) from the triangle.
    # If G is edge-critical, G-e must have diameter > 2 or be disconnected.
    # If G is complete graph K_16, removing an edge leaves diameter 2. Not critical.
    # It is a known result that edge-critical diameter-2 graphs are triangle-free?
    # Actually, consider a graph formed by adding edges to K_{8,8}.
    # If we add an edge within A, say (a1, a2).
    # Now (a1, a2) is an edge.
    # Is (a1, a2) critical?
    # Remove (a1, a2).
    # dist(a1, a2) in G-e: a1 is connected to all B. a2 is connected to all B.
    # Common neighbors: all of B.
    # So dist(a1, a2) = 2.
    # Does the diameter increase?
    # The diameter of G is 2.
    # Removing (a1, a2) might not increase the diameter if all other pairs are still distance <= 2.
    # In K_{8,8} + (a1, a2), the diameter is 2.
    # If we remove (a1, a2), the graph is K_{8,8} minus nothing? No, it's K_{8,8} with (a1, a2) removed.
    # Since (a1, a2) was the only intra-partition edge added, the rest is K_{8,8}.
    # In K_{8,8}, diameter is 2.
    # So G-e has diameter 2.
    # Thus (a1, a2) is NOT critical.
    # Therefore, any edge added to K_{8,8} is not critical.
    # So K_{8,8} is the maximum size.

    result = []
    for i in range(half):
        for j in range(half, n):
            result.append([i, j])

    if time.time() > deadline:
        return result

    return result
# EVOLVE-BLOCK-END
