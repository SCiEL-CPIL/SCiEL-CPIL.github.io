from collections import deque
import time

def _bfs_diameter(adj):
    """Return the diameter of the graph given by adjacency lists.
    Returns float('inf') if the graph is disconnected."""
    n = len(adj)
    diam = 0
    for start in range(n):
        dist = [-1] * n
        dist[start] = 0
        q = deque([start])
        while q:
            u = q.popleft()
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    q.append(v)
        for d in dist:
            if d == -1:
                return float('inf')
            if d > diam:
                diam = d
    return diam

def construct():
    """
    Return edges of a dense graph on vertices 0..15 with diameter exactly 2,
    where deleting any single edge increases the diameter above 2 or disconnects the graph.
    
    Strategy:
    1. Start with K_{8,8} (64 edges), which is known to satisfy the criticality condition.
       In K_{8,8}, removing any edge (u,v) makes dist(u,v) = 3, so diameter becomes 3.
    2. Attempt to add edges within the two partitions (intra-partition edges) while
       preserving the criticality property.
       
    However, adding intra-partition edges generally breaks the criticality of existing
    inter-partition edges. For example, if we add edge (u1, u2) in partition A, then
    for an inter-partition edge (u1, v), removing it might still leave a path u1-u2-...-v
    of length 2 if u2 is connected to v. Since all u in A are connected to all v in B,
    u2 is connected to v. So dist(u1, v) becomes 2. Thus the edge (u1, v) is NOT critical
    because the diameter remains 2 (other pairs might still have dist 2, and u1-v is 2).
    Wait, the condition is: "Deleting EACH individual edge must increase its diameter above two or disconnect it."
    If we add intra-partition edges, the graph is no longer bipartite.
    Let's check if a graph with intra-partition edges can have more edges.
    
    Actually, the problem asks to MAXIMIZE edge count.
    K_{8,8} has 64 edges.
    Is it possible to have a graph with > 64 edges satisfying the condition?
    
    Let's analyze the condition:
    For every edge (u, v) in E, G - (u, v) must have diameter > 2 or be disconnected.
    This implies that for every edge (u, v), there is no path of length 2 between u and v in G - (u, v).
    i.e., N(u) \ {v} and N(v) \ {u} must be disjoint.
    If N(u) \ {v} and N(v) \ {u} intersect, say at w, then u-w-v is a path of length 2.
    Then in G - (u, v), dist(u, v) = 2.
    Does this mean the diameter of G - (u, v) is > 2?
    Not necessarily. If the diameter of G is 2, and we remove an edge that still has a distance-2 path,
    the diameter of the new graph could still be 2 (if all other pairs are still within distance 2).
    The condition requires that the diameter *increases* above 2.
    So if dist(u, v) in G - (u, v) is 2, the diameter of G - (u, v) is at least 2.
    But is it strictly greater than 2?
    If G is a complete graph K_n, removing an edge (u, v) leaves dist(u, v) = 2.
    The diameter of K_n - (u, v) is 2 (for n >= 4).
    So K_n does NOT satisfy the condition.
    
    The condition "diameter increases above 2" means the new diameter must be >= 3.
    This requires that for some pair (x, y), dist_G-(u,v)(x, y) >= 3.
    Usually, this is the pair (u, v) itself if dist(u, v) becomes 3.
    If dist(u, v) becomes 2, then we need some other pair to become distance 3.
    
    In K_{8,8}, for any edge (u, v), N(u) \ {v} = B \ {v} and N(v) \ {u} = A \ {u}.
    These are disjoint (one is in B, one is in A).
    So there is no path of length 2 between u and v.
    The shortest path is length 3: u - b' - a' - v.
    So dist(u, v) = 3.
    Thus diameter of G - (u, v) is 3.
    Condition satisfied.
    
    Can we add edges?
    Suppose we add an edge (u1, u2) where u1, u2 in A.
    Consider edge (u1, v) where v in B.
    In G - (u1, v), is there a path of length 2 between u1 and v?
    N(u1) in G includes A\{u1} (if we added enough intra-A edges) and B.
    Specifically, if (u1, u2) is an edge, u2 is in N(u1).
    Is u2 in N(v)? Yes, because (u2, v) is an edge in K_{8,8}.
    So u1 - u2 - v is a path of length 2.
    So dist(u1, v) = 2 in G - (u1, v).
    Does the diameter increase?
    The diameter of G is 2.
    In G - (u1, v), dist(u1, v) = 2.
    Are there any pairs with distance > 2?
    If we only added one intra-edge, the graph is very dense.
    Likely the diameter remains 2.
    If the diameter remains 2, the condition "diameter increases above 2" is FAILED.
    
    Therefore, adding intra-partition edges likely breaks the criticality of inter-partition edges.
    
    What if we construct a different graph?
    The condition implies that for every edge (u, v), u and v are "far" apart in G - (u, v).
    Specifically, they must be at distance >= 3.
    This implies that u and v have no common neighbors in G - (u, v).
    i.e., (N(u) \ {v}) ∩ (N(v) \ {u}) = ∅.
    This means N(u) ∩ N(v) ⊆ {u, v} is false, rather N(u) ∩ N(v) must be empty?
    No. N(u) and N(v) are sets of vertices.
    If w is in N(u) and w is in N(v), then w is a common neighbor.
    If w != u and w != v (which is true for simple graphs), then u-w-v is a path of length 2.
    So if N(u) ∩ N(v) is non-empty, dist(u, v) <= 2 in G - (u, v).
    For the diameter to increase to >= 3, it is sufficient that dist(u, v) = 3.
    If dist(u, v) = 2, the diameter might still be 2.
    To guarantee diameter >= 3, we need dist(u, v) >= 3.
    This requires N(u) ∩ N(v) = ∅.
    
    So the condition "every edge is critical (diameter increases)" is satisfied if for every edge (u, v), N(u) ∩ N(v) = ∅.
    This means the graph is triangle-free?
    If (u, v) is an edge, and N(u) ∩ N(v) = ∅, then there is no w connected to both u and v.
    So no triangle u-v-w-u.
    Thus the graph must be triangle-free.
    
    If the graph is triangle-free, does every edge satisfy the condition?
    If G is triangle-free and has diameter 2.
    Remove edge (u, v).
    Since no common neighbors, dist(u, v) in G - (u, v) is at least 3.
    If the graph is connected, dist(u, v) is finite.
    So dist(u, v) >= 3.
    Thus diameter of G - (u, v) >= 3.
    So the condition is satisfied.
    
    So we need a triangle-free graph on 16 vertices with diameter 2 that maximizes edges.
    By Mantel's Theorem, the maximum number of edges in a triangle-free graph on n vertices is floor(n^2 / 4).
    For n = 16, max edges = 16^2 / 4 = 64.
    The extremal graph is K_{8,8}.
    
    So K_{8,8} is the optimal solution.
    Any other triangle-free graph with diameter 2 has <= 64 edges.
    If we allow non-triangle-free graphs, we might have more edges, but then some edges would have common neighbors, leading to dist(u, v) = 2 upon removal, potentially keeping diameter at 2.
    As argued above, if dist(u, v) = 2 in G - (u, v), the diameter likely stays 2 in dense graphs.
    To be safe, the triangle-free condition is a sufficient condition for criticality.
    And Mantel's theorem bounds the edges for triangle-free graphs.
    
    Therefore, the maximum edge count is 64, achieved by K_{8,8}.
    """
    n = 16
    half = 8
    result = []
    for u in range(half):
        for v in range(half, n):
            result.append([u, v])
    return result
