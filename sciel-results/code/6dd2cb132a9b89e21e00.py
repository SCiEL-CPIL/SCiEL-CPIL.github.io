"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def construct():
    """
    Construct a dense graph on vertices 0..15 with diameter exactly two,
    where every edge is essential (deleting any edge increases diameter above 2
    or disconnects the graph). Maximize edge count.
    
    Strategy: The unique extremal triangle-free graph on 16 vertices with maximum edges
    is the complete bipartite graph K_{8,8} with 64 edges. Every edge is essential because:
    - Removing any edge from K_{8,8} disconnects the two vertices (since there's no alternative path
      of length 1, and any path of length 2 would require a common neighbor, which doesn't exist
      in a bipartite graph for vertices in the same part).
    
    Actually, let me reconsider. In K_{8,8}, if we remove edge (u,v) where u is in part A and v is in part B:
    - u and v are still connected via paths of length 3 (u -> another vertex in B -> another vertex in A -> v),
      so the graph remains connected.
    - But the distance between u and v becomes 3, not 2. So the diameter might still be 2 if there are other pairs
      at distance 2, but the specific pair (u,v) is now at distance 3.
    
    Wait, the requirement is that deleting EACH individual edge must increase its diameter above two OR disconnect it.
    So if removing edge (u,v) makes dist(u,v) = 3, then the diameter becomes at least 3, which is above 2. Good.
    
    But is K_{8,8} the maximum? Let's think about other structures.
    
    Actually, for a graph where every edge is critical for diameter 2, we need that for every edge (u,v),
    there is no alternative path of length 2 between u and v. This means u and v have no common neighbor.
    This is equivalent to saying the graph is triangle-free AND for every edge, the endpoints have no common neighbor.
    
    Wait, that's not quite right. If (u,v) is an edge, and they have a common neighbor w, then there's a path u-w-v of length 2,
    so removing edge (u,v) would still leave dist(u,v) <= 2. So for every edge (u,v), u and v must have no common neighbor.
    This means the graph is "2-triangle-free" or has girth at least 5.
    
    So we need a graph with girth at least 5, diameter exactly 2, and as many edges as possible on 16 vertices.
    
    The maximum number of edges in a graph with girth at least 5 on n vertices is bounded. For n=16,
    the extremal number ex(16, {C3, C4}) is the maximum edges in a C3-free and C4-free graph.
    
    Actually, we need girth >= 5, so no C3 and no C4.
    
    The maximum number of edges in a graph with girth at least 5 on 16 vertices:
    - For a (5,3)-cage or similar, but let's compute.
    - The Moore bound for girth 5 and degree d: n >= 1 + d + d(d-1) = d^2 + 1.
      For d=3: n >= 10. For d=4: n >= 17. So with n=16, max degree is at most 3 for regular graphs,
      but we can have non-regular.
    
    Let me just construct a candidate and verify.
    
    Actually, let me reconsider the problem. The key insight from the knowledge is that K_{8,8} scored 4.267,
    which is 64/15 = 4.267. So the seed had 15 edges (a star), and K_{8,8} has 64 edges.
    
    But wait, does K_{8,8} actually satisfy the condition? Let's check:
    - Diameter of K_{8,8} is 2 (any two vertices in the same part have distance 2 via any vertex in the other part).
    - Removing edge (u,v) where u in A, v in B: u and v are now at distance 3 (u -> u' in B -> v' in A -> v, wait no).
      Actually in K_{8,8}, after removing (u,v), u is still connected to all other vertices in B, and v is still connected to all other vertices in A.
      So dist(u,v) = 3: u -> w (w in B, w != v) -> x (x in A, x != u) -> v. Wait, that's length 3.
      But is there a shorter path? u is connected to all B except v. v is connected to all A except u.
      So any path from u to v must go u -> b -> a -> v where b in B, a in A. That's length 3.
      So dist(u,v) = 3 after removing (u,v). Thus diameter becomes 3 > 2. Good.
    
    So K_{8,8} works and has 64 edges. Can we do better?
    
    The question is: what's the maximum number of edges in a graph on 16 vertices with diameter 2
    where every edge is critical?
    
    Let me think about this differently. We want a graph where:
    1. Diameter is exactly 2.
    2. For every edge (u,v), there is no path of length 2 between u and v (otherwise removing (u,v) wouldn't increase dist(u,v) above 2).
    
    Condition 2 means: for every edge (u,v), N(u) ∩ N(v) = ∅. This is the same as saying the graph has no triangles and no 4-cycles containing any edge... wait, no.
    
    If u and v are adjacent, and they have a common neighbor w, then u-w-v is a path of length 2. So we need N(u) ∩ N(v) = ∅ for every edge (u,v).
    This means: no triangles (since a triangle u-v-w-u would give u and v sharing neighbor w).
    Also, it means no 4-cycles of the form u-w-v-x-u where (u,v) is an edge... wait, no.
    
    Actually, N(u) ∩ N(v) = ∅ for edge (u,v) just means no common neighbor, which forbids triangles. It doesn't directly forbid 4-cycles.
    
    But wait, if there's a 4-cycle u-a-v-b-u, then u and v share... no, u's neighbors are a and b, v's neighbors are a and b. So N(u) ∩ N(v) = {a, b} ≠ ∅. So 4-cycles are also forbidden!
    
    So the graph must have girth at least 5 (no C3, no C4).
    
    So we need a graph on 16 vertices with girth at least 5, diameter exactly 2, maximizing edges.
    
    The maximum number of edges in a graph with girth at least 5 on n vertices is known as the extremal number ex(n, {C3, C4}).
    
    For n=16, let me try to construct such a graph.
    
    A (5,3)-cage has 10 vertices and 15 edges. We need 16 vertices.
    
    Let me try to use a known construction or just do a search.
    
    Actually, let me just go with K_{8,8} since it's known to work and has 64 edges. If there's a better graph, it would need more than 64 edges, but a graph with girth at least 5 on 16 vertices can't have too many edges.
    
    The average degree would be 2*E/16 = E/8. If E > 64, avg degree > 8. But with girth 5, the number of vertices at distance 2 from a vertex of degree d is at least d(d-1). So for d=8, we need at least 8*7=56 vertices at distance 2, but we only have 15 other vertices. Contradiction.
    
    So max degree is bounded. For vertex of degree d, need 1 + d + d(d-1) <= 16, so d(d-1) <= 15, d <= 4 (since 4*3=12 <= 15, 5*4=20 > 15).
    
    So max degree is 4? That gives at most 16*4/2 = 32 edges. But K_{8,8} has degree 8 for every vertex and 64 edges. Contradiction!
    
    Wait, I made an error. In K_{8,8}, two adjacent vertices u (in A) and v (in B) have N(u) = B and N(v) = A. So N(u) ∩ N(v) = ∅. Good, no common neighbor. But there are 4-cycles! For example, u1-a1-v1-b1-u1 where u1, v1 in A and a1, b1 in B. So K_{8,8} has 4-cycles.
    
    But the condition is only that for each EDGE (u,v), N(u) ∩ N(v) = ∅. In K_{8,8}, for edge (u,v) with u in A, v in B, N(u) = B, N(v) = A, so intersection is empty. The 4-cycles don't violate this condition because the condition is about adjacent vertices sharing a neighbor, not about non-adjacent vertices.
    
    So K_{8,8} is valid! And it has 64 edges.
    
    Can we have more than 64 edges? Total possible edges on 16 vertices is C(16,2) = 120.
    
    Let me think... if we add any edge within A or within B to K_{8,8}, say (u1, u2) where u1, u2 in A. Then N(u1) includes all of B plus u2. N(u2) includes all of B plus u1. So N(u1) ∩ N(u2) includes all of B, which is non-empty. So edge (u1, u2) would NOT be critical (removing it wouldn't increase dist(u1,u2) above 2, since they still have common neighbors in B). So we can't add such edges.
    
    So K_{8,8} with 64 edges is likely optimal. Let me just return that.
    
    """
    n = 16
    # K_{8,8}: partition into A = {0,...,7} and B = {8,...,15}
    edges = []
    for u in range(8):
        for v in range(8, 16):
            edges.append([u, v])
    
    return edges
# EVOLVE-BLOCK-END
