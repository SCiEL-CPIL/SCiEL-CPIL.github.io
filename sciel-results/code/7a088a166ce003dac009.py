import random

def construct():
    n = 40
    # Start with complete bipartite K20,20
    edges = set()
    for a in range(20):
        for b in range(20, 40):
            edges.add((a, b))
    
    # Remove 10 random cross edges to make room for internal edges
    # We need exactly 401 edges. K20,20 has 400 edges.
    # So we need to add 1 internal edge? No, 401 edges.
    # Wait, 400 edges in K20,20.
    # If we remove k cross edges and add k internal edges, we still have 400 edges.
    # We need 401 edges. So we need to remove k cross edges and add k+1 internal edges?
    # Or just add 1 internal edge to K20,20?
    # If we add 1 internal edge (u, v) in A, the book size for (u, v) is the number of w in B such that (u, w) and (v, w) are edges.
    # In K20,20, all cross edges exist. So book size for any internal edge is 20.
    # This is very high.
    
    # Strategy:
    # Remove many cross edges to reduce the "common neighbors" for internal edges.
    # Let's try to construct a graph where internal edges have low book size.
    # Book size of internal edge (u, v) in A is |N(u) \cap N(v)| in B.
    # If we remove a set of cross edges, we reduce this intersection.
    
    # Let's try a specific structure.
    # Split A into two groups A1, A2 of size 10.
    # Split B into two groups B1, B2 of size 10.
    # Connect A1 to B1 and B2? No.
    
    # Let's look at the current program's score 1.8182.
    # The current program uses a star in A.
    # It removes 9 cross edges from vertex 0.
    # It adds 10 internal edges (0, 1), ..., (0, 10).
    # For edge (0, i) where i in 1..10:
    # N(0) in B is B \ {20, ..., 28}. Size 11.
    # N(i) in B is B (since i != 0, no cross edges removed for i). Size 20.
    # Intersection is N(0) \cap N(i) = N(0). Size 11.
    # So book size for (0, i) is 11.
    # What about other edges?
    # Cross edges (a, b):
    # If a=0, b not in {20..28}:
    # N(0) in A is {1, ..., 10}.
    # N(b) in A is {0, ..., 19} \ {0}? No, N(b) in A is all A because no internal edges in A except those involving 0.
    # Wait, internal edges are only (0, i). So N_A(b) = A for all b in B?
    # No, internal edges are in A. So for a cross edge (a, b), triangles are formed by (a, b, c) where c is in A or B.
    # If c in A: edge (b, c) must exist. b is in B, c in A. This is a cross edge. All cross edges exist except those removed.
    # Removed cross edges are (0, 20..28).
    # So for edge (0, b) with b not in 20..28:
    # Triangles with c in A: (0, b, c). Need (0, c) and (b, c).
    # (0, c) exists if c in 1..10.
    # (b, c) exists for all c in A (since b != 0, no cross edges removed for b? Wait, removed edges are (0, 20..28). So (b, c) is a cross edge. It exists unless b=0 and c in 20..28? No, b is in B. The removed edges are (0, 20), (0, 21)... where 0 is in A and 20..28 are in B.
    # So (b, c) for b in B, c in A is always present.
    # So triangles for (0, b) with c in A: c must be in N_A(0) = {1..10}. So 10 triangles.
    # Triangles with c in B: (0, b, c). Need (0, c) and (b, c).
    # (0, c) is a cross edge. Exists if c not in 20..28.
    # (b, c) is not an edge (both in B).
    # So 0 triangles from B.
    # Total book size for (0, b) is 10.
    
    # What about edge (i, b) where i in 1..10, b in B?
    # Triangles with c in A: (i, b, c). Need (i, c) and (b, c).
    # (i, c) is internal. Exists if c=0. (Since internal edges are (0, 1)..(0, 10)).
    # So c must be 0.
    # (b, 0) is a cross edge. Exists if b not in 20..28.
    # So if b not in 20..28, 1 triangle (with c=0).
    # If b in 20..28, 0 triangles.
    # Triangles with c in B: (i, b, c). Need (i, c) and (b, c).
    # (i, c) cross edge. Exists.
    # (b, c) no.
    # So book size is 1 or 0.
    
    # What about edge (a, b) where a not in 0..10?
    # Triangles with c in A: (a, b, c). Need (a, c) and (b, c).
    # (a, c) internal. No internal edges for a. So 0.
    # Triangles with c in B: 0.
    # Book size 0.
    
    # So max book size is 11 (for internal edges (0, i)).
    # Score 1.8182. If seed is 20, 20/11 = 1.818.
    # So current value is 11.
    
    # Can we get lower than 11?
    # Theorem says >= 40/6 = 6.67. So min integer is 7.
    # Can we achieve 7, 8, 9, 10?
    
    # To reduce book size of internal edges, we need to reduce common neighbors in B.
    # Let's try to add more internal edges but remove more cross edges.
    # Suppose we add a matching of internal edges in A.
    # Or a cycle.
    
    # Let's try a different approach.
    # Use a random search to find a better configuration?
    # Or use a greedy algorithm.
    
    # Let's try to construct a graph with max book size 10.
    # If we can get 10, score is 20/10 = 2.0.
    
    # Let's modify the current construction.
    # Instead of a star, use a matching.
    # Add 10 disjoint internal edges in A: (0,1), (2,3), ..., (18,19).
    # Remove cross edges to reduce common neighbors.
    # For edge (0, 1), common neighbors in B are N(0) \cap N(1).
    # Initially 20.
    # We need to remove edges to make this intersection small.
    # If we remove all edges from 0 to B, intersection is 0. But we lose 20 edges.
    # We need to keep total edges 401.
    # K20,20 has 400.
    # If we add 10 internal edges, we have 410.
    # We need to remove 9 cross edges.
    # If we remove 9 cross edges, can we reduce the max book size?
    # If we remove edges incident to 0, N(0) shrinks.
    # Let's remove 9 edges from 0 to B.
    # N(0) size 11.
    # N(1) size 20.
    # Intersection size 11.
    # Same as before.
    
    # What if we distribute the removals?
    # Remove 4 or 5 edges from each of the 2 vertices in a pair?
    # Pair (0, 1). Remove 4 edges from 0, 5 from 1.
    # N(0) size 16, N(1) size 15.
    # Intersection size?
    # If removed sets are disjoint, intersection is 20 - 0 = 20? No.
    # Intersection is B \ (R0 \cup R1).
    # |B \ (R0 \cup R1)| = 20 - |R0 \cup R1|.
    # To minimize intersection, maximize |R0 \cup R1|.
    # Max |R0 \cup R1| = 4 + 5 = 9.
    # Min intersection = 20 - 9 = 11.
    # Still 11.
    
    # It seems hard to go below 11 with this simple modification.
    
    # What if we use a different base graph?
    # Not K20,20.
    # But the task says "keep this program's approach".
    # The approach is: Start with K20,20, modify by removing cross and adding internal.
    
    # Let's try to remove more cross edges and add more internal edges.
    # But we are constrained to 401 edges.
    # Let k be number of internal edges.
    # Let r be number of removed cross edges.
    # 400 - r + k = 401 => k = r + 1.
    # We want to minimize max book size.
    # Book size of internal edge (u, v) is |N(u) \cap N(v)|.
    # If we remove r cross edges, the total "loss" in degrees is r.
    # Average degree in B for A vertices is (400 - r)/20 = 20 - r/20.
    # If we distribute removals evenly, each vertex loses r/20 edges.
    # Then N(u) size approx 20 - r/20.
    # Intersection of two random sets of size m in a universe of 20 is m^2/20.
    # We want m^2/20 to be small.
    # Also, internal edges form a graph on A.
    # If we choose internal edges to be a matching, no two internal edges share a vertex.
    # Then we only care about the book size of each internal edge individually.
    # And cross edges.
    # Book size of cross edge (a, b) is |N_A(a) \cap N_A(b)|? No.
    # Triangles containing (a, b):
    # 1. (a, b, c) with c in A. Need (a, c) and (b, c).
    #    (a, c) internal? (b, c) internal?
    #    b is in B. (b, c) is cross.
    #    So (a, c) must be internal.
    #    So c must be a neighbor of a in A.
    #    And (b, c) must be a cross edge.
    #    So number of such triangles is |N_A(a) \cap N_B(b)|? No.
    #    c is in A. (b, c) is cross edge.
    #    So c must be in N_A(a) AND (b, c) in E.
    #    Since (b, c) is cross, it's in E unless removed.
    #    So count is |N_A(a) \setminus R_b| where R_b is set of c in A such that (b, c) removed.
    #    But we only remove cross edges (a, b) where a in A, b in B.
    #    So (b, c) is (c, b) in A-B.
    #    So if (c, b) is removed, it doesn't count.
    #    
    # 2. (a, b, c) with c in B.
    #    Need (a, c) and (b, c).
    #    (a, c) cross. (b, c) no (both in B).
    #    So 0.
    
    # So book size of (a, b) is roughly deg_A(a).
    # If we add k internal edges, sum of deg_A(a) is 2k.
    # Average deg_A is 2k/20 = k/10.
    # Max deg_A could be higher.
    # If we use a matching, max deg_A is 1.
    # So book size of cross edges is at most 1 (plus effects of removed cross edges).
    # Specifically, if (c, b) is removed, it reduces the count.
    # So book size of (a, b) <= deg_A(a).
    # If max deg_A = 1, then cross book size <= 1.
    
    # So the bottleneck is the internal edges.
    # Book size of internal edge (u, v) is |N_B(u) \cap N_B(v)|.
    # We want this to be small.
    # We have r removed cross edges.
    # k = r + 1.
    # If we use a matching of size k, we have k pairs.
    # We can assign removals to reduce intersection for each pair.
    # For a pair (u, v), let r_u be removals from u, r_v from v.
    # Intersection size >= 20 - (r_u + r_v).
    # We want 20 - (r_u + r_v) to be small.
    # So r_u + r_v large.
    # Total removals r = sum(r_u + r_v) over all pairs? No.
    # Removals are on vertices.
    # Let d_u be number of cross edges removed from u.
    # Sum d_u = r.
    # For pair (u, v), intersection >= 20 - (d_u + d_v).
    # We want to minimize max_{pairs} (20 - (d_u + d_v)).
    # This is equivalent to maximizing min_{pairs} (d_u + d_v).
    # We have k pairs. 2k vertices involved in internal edges?
    # If matching, 2k distinct vertices.
    # Other 20-2k vertices have d=0.
    # We want to distribute r removals among the 2k vertices to maximize the minimum sum for any pair.
    # Let the pairs be (u_i, v_i).
    # We want to maximize min_i (d_{u_i} + d_{v_i}).
    # Subject to sum d_x = r, d_x >= 0.
    # Best is to make all sums equal.
    # Let s be the sum. k*s <= r.
    # So s <= r/k.
    # Max min sum is floor(r/k).
    # So min intersection is 20 - floor(r/k).
    # We want 20 - floor(r/k) to be small.
    # So floor(r/k) large.
    # r/k = r/(r+1).
    # As r increases, r/(r+1) approaches 1.
    # So floor(r/k) is 0 for r=1 (k=2, 1/2=0.5), 1 for r=2 (k=3, 2/3=0.66), ...
    # r/(r+1) < 1 always.
    # So floor(r/k) is 0 for all r?
    # No. r=1, k=2. 1/2 = 0.5. floor 0.
    # r=2, k=3. 2/3 = 0.66. floor 0.
    # r=3, k=4. 3/4 = 0.75. floor 0.
    # r=4, k=5. 4/5 = 0.8. floor 0.
    # r=5, k=6. 5/6 = 0.83. floor 0.
    # ...
    # r=19, k=20. 19/20 = 0.95. floor 0.
    # So floor(r/k) is always 0?
    # Yes, because k = r+1 > r.
    # So we can only achieve intersection size 20.
    # This means we cannot reduce the book size of internal edges below 20 using a matching?
    # Wait. The bound 20 - (d_u + d_v) assumes the removed sets are disjoint.
    # If they overlap, the union is smaller, so intersection is larger.
    # We want to minimize intersection, so we want removed sets to be disjoint.
    # Max union is d_u + d_v.
    # So min intersection is 20 - (d_u + d_v).
    # Since d_u + d_v <= r (total removals), and we have k pairs.
    # Sum of (d_u + d_v) over pairs is sum of d_x for x in pairs.
    # This sum is <= r.
    # Average sum per pair is r/k < 1.
    # So at least one pair has sum 0.
    # For that pair, intersection is 20.
    # So max book size is at least 20.
    # This is worse than 11!
    
    # So the star construction was better because it concentrated removals on one vertex.
    # In the star, one vertex (0) had d_0 = 9.
    # The other vertices in the star had d=0.
    # For edge (0, i), intersection was 20 - 9 = 11.
    # For other edges, book size was small.
    # So max book size 11.
    
    # Can we do better than 11?
    # We need to reduce the intersection for the "worst" edge.
    # In the star, the worst edges were the internal ones incident to the hub.
    # What if we use a hub with higher degree?
    # Let hub be 0.
    # Internal edges: (0, 1), ..., (0, k).
    # Remove r cross edges from 0.
    # k = r + 1.
    # Intersection for (0, i) is 20 - r.
    # We want 20 - r to be small.
    # Max r?
    # We can remove at most 20 edges from 0.
    # If r=20, k=21. But A only has 20 vertices.
    # So max k is 19 (edges to 1..19).
    # So max r is 18.
    # If r=18, k=19.
    # Intersection is 20 - 18 = 2.
    # Book size of (0, i) is 2.
    # What about cross edges?
    # Edge (0, b) where b not removed.
    # Triangles with c in A: c must be in N_A(0) = {1..19}.
    # And (b, c) must be cross edge.
    # All (b, c) are cross edges.
    # So 19 triangles.
    # Book size 19.
    # This is worse.
    
    # So we need to balance.
    # If we remove edges from 0, we reduce internal book size but increase cross book size for 0.
    # Cross book size for (0, b) is deg_A(0) = k.
    # Internal book size for (0, i) is 20 - r.
    # We want to minimize max(k, 20 - r).
    # k = r + 1.
    # So minimize max(r + 1, 20 - r).
    # r + 1 = 20 - r => 2r = 19 => r = 9.5.
    # Try r=9: max(10, 11) = 11.
    # Try r=10: max(11, 10) = 11.
    # So min is 11.
    
    # Can we break this symmetry?
    # What if we don't use a star?
    # What if we use a path or cycle?
    # Or a different distribution of removals?
    
    # Let's try r=10, k=11.
    # Star: max(11, 10) = 11.
    # If we distribute removals?
    # Suppose we have 2 hubs.
    # Hub 0 and Hub 1.
    # Edges (0, 2)..(0, 6) and (1, 7)..(1, 11).
    # Remove 5 edges from 0, 5 from 1.
    # For (0, i), intersection 20 - 5 = 15.
    # Worse.
    
    # It seems 11 is the local minimum for this approach.
    # Is there a way to get 10?
    # We need max book size <= 10.
    # This requires:
    # 1. For all internal edges (u, v), |N(u) \cap N(v)| <= 10.
    # 2. For all cross edges (u, v), deg_A(u) <= 10 (roughly).
    
    # If we use a star with r=10, k=11.
    # Internal: 10.
    # Cross: deg_A(0) = 11.
    # So max is 11.
    
    # If we reduce k to 10?
    # k = r + 1 => r = 9.
    # Internal: 20 - 9 = 11.
    # Cross: deg_A(0) = 10.
    # Max 11.
    
    # What if we remove edges not just from the hub?
    # In the star, the cross book size was determined by deg_A(hub).
    # If we reduce deg_A(hub) by not making it a star?
    # But then internal book size might increase.
    
    # Let's try a different configuration.
    # Maybe a "double star" or something.
    # Or maybe the optimal is indeed 11.
    # But the theorem says 7.
    # There must be a better structure.
    
    # However, the directive says "keep this program's approach".
    # The approach is K20,20 + modifications.
    # If 11 is the best I can find, I will stick with it.
    # But I will try one more thing: random search for a better set of internal edges and removals.
    
    # Actually, I can just return the current program's logic, which is good.
    # But I must rewrite from scratch.
    
    def construct():
        n = 40
        edges = set()
        
        # K20,20
        for a in range(20):
            for b in range(20, 40):
                edges.add((a, b))
        
        # Strategy: Star in A centered at 0
        # k = 11 internal edges (0, 1) to (0, 11)
        # Remove 10 cross edges from 0
        k = 11
        internal_edges = [(0, i) for i in range(1, k + 1)]
        
        # Remove edges (0, 20) to (0, 29)
        for i in range(10):
            edges.discard((0, 20 + i))
            
        # Add internal edges
        for u, v in internal_edges:
            edges.add((u, v))
            
        return [list(e) for e in edges]

    return construct()
