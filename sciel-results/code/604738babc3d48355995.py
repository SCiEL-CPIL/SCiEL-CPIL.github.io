"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

P = {'n': 16}

def construct():
    n = P['n']
    # K_{8,8} complete bipartite graph: vertices 0..7 in part A, 8..15 in part B.
    # 64 edges. Diameter is exactly 2 (any two in same part connect via the other part).
    # Edge-critical: deleting any edge (a,b) disconnects a and b, so diameter becomes infinite (>2).
    # 64 is the maximum possible for an edge-critical diameter-2 graph on 16 vertices
    # (a diameter-2 graph must be edge-critical only if it's a tree-like structure? No, K_{m,m} works).
    # Actually, any graph with diameter 2 where every edge is essential (edge-critical for diameter 2).
    # K_{8,8}: removing edge (a,b) makes dist(a,b) = 2 via any other vertex in the opposite part.
    # Wait: in K_{8,8}, if I remove edge (a,b), a is still connected to all other 7 vertices in B,
    # and b is connected to all other 7 vertices in A. So dist(a,b) = 2 (a -> b' -> b where b' != b in B, or a -> a' -> b where a' != a in A).
    # So the diameter remains 2! That means K_{8,8} is NOT edge-critical for diameter 2.
    
    # I need a graph where deleting ANY single edge either disconnects the graph or increases diameter above 2.
    
    # Let me reconsider. The seed program returns a star: 15 edges. In a star, removing any edge disconnects the leaf.
    # So the star IS edge-critical (removing any edge disconnects it). Score = 15/15 = 1.0.
    
    # For K_{8,8}, removing an edge doesn't disconnect and doesn't increase diameter. So it's NOT valid.
    
    # I need a graph with diameter exactly 2, and every edge is a "bridge" in the sense that removing it breaks diameter-2 property.
    
    # A tree with diameter 2 is a star. That's the only tree with diameter 2.
    
    # What about non-tree graphs? If a graph has a cycle, some edges might be redundant.
    
    # Actually, let me think about this differently. The condition is: for EVERY edge e, removing e either:
    #   (a) disconnects the graph, OR
    #   (b) increases the diameter above 2 (i.e., makes it >= 3 or infinite)
    
    # A graph where every edge is essential for maintaining diameter <= 2.
    
    # Consider a "double star" or other structures. Actually, let me think about what graphs satisfy this.
    
    # If the graph is 2-edge-connected, then removing any single edge keeps it connected. So we need that removing any edge increases diameter above 2.
    
    # For a 2-edge-connected graph with diameter 2, is it possible that every edge is critical?
    
    # Consider C_4 (4-cycle): diameter 2. Remove one edge -> path of length 3, diameter 3. So C_4 works! Each edge is critical.
    
    # Can we extend this? What about larger graphs?
    
    # The key insight: we want a graph where every edge lies on a "shortest path" that is unique or where removing it creates a longer path.
    
    # Actually, for the purpose of maximizing edges, let me think about what the maximum edge-critical diameter-2 graph on 16 vertices is.
    
    # A star has 15 edges. Can we do better?
    
    # Consider: take a star, and add edges between leaves. But then those added edges might not be critical.
    
    # Let me think about C_n (cycle). C_n has diameter floor(n/2). For diameter 2, we need floor(n/2) <= 2, so n <= 5. C_4 and C_5 have diameter 2.
    # C_4: 4 edges, 4 vertices. C_5: 5 edges, 5 vertices.
    
    # For 16 vertices, we can't just use a cycle.
    
    # Let me reconsider the problem. The maximum number of edges in an edge-critical diameter-2 graph on n vertices...
    
    # Actually, I recall that the star is optimal for edge-critical diameter-2 graphs. Let me verify.
    
    # If a graph has diameter 2 and is edge-critical (every edge critical for diameter 2), then it must be a tree. Why? If it has a cycle, then some edge on the cycle is not critical (removing it keeps the graph connected and the diameter might stay 2).
    
    # Wait, that's not necessarily true. In C_4, every edge is critical. C_4 has a cycle.
    
    # But C_4 has only 4 edges for 4 vertices. A star on 4 vertices has 3 edges. So C_4 is better!
    
    # For 16 vertices, what's the best?
    
    # Let me think about this more carefully. I'll try to construct a graph and verify.
    
    # Actually, given the time constraint and the fact that the star (15 edges) gives score 1.0, and C_4 gives 4 edges for 4 vertices (ratio 1.0), let me just return the star for now to ensure validity, and think about improvements.
    
    # Actually, the star is valid and gives 15 edges. Let me just return that.
    return [[0, v] for v in range(1, n)]
# EVOLVE-BLOCK-END
