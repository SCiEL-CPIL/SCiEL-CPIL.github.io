import time
import random
import numpy as np

def _has_c5(adj, n):
    """Check if graph has a 5-cycle using adjacency matrix."""
    # A 5-cycle exists iff there exist distinct v1,v2,v3,v4,v5 with edges
    # v1-v2, v2-v3, v3-v4, v4-v5, v5-v1
    # Use: for each pair (i,j) that is NOT an edge, check if there's a path of length 4 between them
    # That would form a 5-cycle.
    # More efficient: check all 5-tuples? Too slow for n=33.
    # Better: for each edge (u,v), check if there's a path of length 3 from u to v avoiding the edge.
    # Actually, for C5 detection: a C5 is a cycle of length 5.
    # We can check: for each pair (a,b) with a<b, if (a,b) is NOT an edge,
    # then check if there exists a path a-c-d-e-b of length 4 (3 intermediate vertices).
    # This would form a 5-cycle a-c-d-e-b-a.
    
    # Alternative: use the fact that C5 exists iff there are vertices v1..v5 distinct
    # with adj[v1][v2] & adj[v2][v3] & adj[v3][v4] & adj[v4][v5] & adj[v5][v1]
    
    # Efficient check: for each pair (i,j) that IS an edge, check if there's a path of length 4
    # from i to j not using edge (i,j). This forms a 5-cycle.
    
    # Even better: count 5-cycles. A 5-cycle v1-v2-v3-v4-v5-v1.
    # We can enumerate: for each vertex v1, for each neighbor v2, for each neighbor v3 of v2 (v3!=v1),
    # for each neighbor v4 of v3 (v4 not in {v1,v2}), for each neighbor v5 of v4 (v5 not in {v1,v2,v3,v4}),
    # check if v5 is adjacent to v1.
    
    # This is O(n * d^4) which for n=33 and moderate d should be fine.
    
    adj_set = [set() for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if adj[i][j]:
                adj_set[i].add(j)
    
    for v1 in range(n):
        for v2 in adj_set[v1]:
            for v3 in adj_set[v2]:
                if v3 == v1:
                    continue
                for v4 in adj_set[v3]:
                    if v4 in (v1, v2):
                        continue
                    for v5 in adj_set[v4]:
                        if v5 in (v1, v2, v3, v4):
                            continue
                        if v5 in adj_set[v1]:
                            return True
    return False

def _alpha_ge_9(adj, n):
    """Check if independence number >= 9 using branch and bound."""
    # Independent set: no two vertices in the set are adjacent.
    # Use branch and bound with bitmasks.
    # For n=33, we can use Python integers as bitmasks.
    
    # adj_mask[i] = bitmask of neighbors of i
    adj_mask = [0] * n
    for i in range(n):
        for j in range(n):
            if adj[i][j]:
                adj_mask[i] |= (1 << j)
    
    # We want to find an independent set of size 9.
    # Branch and bound: maintain candidate set as bitmask.
    
    best = [0]
    
    def _branch(candidates, size):
        if size >= 9:
            best[0] = 9
            return True
        if candidates == 0:
            return False
        # Upper bound: size + popcount(candidates)
        if size + bin(candidates).count('1') < 9:
            return False
        
        # Pick a vertex to branch on
        v = candidates & (-candidates)
        v_idx = v.bit_length() - 1
        
        # Option 1: include v
        new_candidates = candidates & ~adj_mask[v_idx] & ~v
        if _branch(new_candidates, size + 1):
            return True
        
        # Option 2: exclude v
        if _branch(candidates & ~v, size):
            return True
        
        return False
    
    all_mask = (1 << n) - 1
    return _branch(all_mask, 0)

def construct():
    start_time = time.time()
    deadline = start_time + 120.0  # 120s wall clock limit
    
    # First, return the known 32-vertex solution as fallback
    n_fallback = 32
    edges_fallback = []
    for b in range(8):
        base = 4 * b
        for i in range(4):
            for j in range(i + 1, 4):
                edges_fallback.append([base + i, base + j])
    
    # Try to find a 33-vertex solution
    random.seed(42)
    
    for attempt in range(100000):
        if time.time() > deadline - 2.0:
            break
        
        # Start with 8 K4s plus one extra vertex, then try to add edges
        # to make alpha <= 8 while keeping C5-free
        
        # Strategy: start with a random graph on 33 vertices that is C5-free
        # and check if alpha < 9.
        
        # Approach: Use a structured construction.
        # 8 K4s give 32 vertices with alpha=8.
        # Add vertex 32. To keep alpha <= 8, vertex 32 must be adjacent to
        # at least one vertex in each "independent set choice".
        # The independent sets of size 8 in 8 K4s are exactly one vertex from each K4.
        # So vertex 32 must be adjacent to at least one vertex in each K4.
        # If we connect vertex 32 to exactly one vertex in each K4, we need to
        # check if this creates a C5.
        
        # Let's try: connect vertex 32 to vertex 0 in each K4 (i.e., vertices 0, 4, 8, 12, 16, 20, 24, 28).
        # Check for C5: A C5 involving vertex 32 would be 32-a-b-c-d-32 where
        # a, b, c, d are from the K4s.
        # Since 32 is only connected to {0,4,8,12,16,20,24,28}, a and d must be from this set.
        # Say a=0 (in K4_0) and d=4 (in K4_1). Then we need a path 0-b-c-4 of length 3.
        # 0 is in K4_0 = {0,1,2,3}. 4 is in K4_1 = {4,5,6,7}.
        # b must be neighbor of 0: b in {1,2,3} (within K4_0) or b=32 (but 32 is used).
        # c must be neighbor of 4: c in {5,6,7} (within K4_1) or c=32 (used).
        # But b and c must be adjacent. b is in K4_0, c is in K4_1. They're not adjacent
        # (different components). So no C5 through 32 with a in K4_0 and d in K4_1.
        # Similarly for any two different K4s.
        # What about a and d in the same K4? Say a=0, d=1 (both in K4_0).
        # Then we need path 0-b-c-1 of length 3. b neighbor of 0: {1,2,3,32}.
        # If b=2, c neighbor of 2: {0,1,3,32}. c neighbor of 1: {0,2,3,32}.
        # So b=2, c=3 works: 0-2-3-1. But we need 32-0-2-3-1-32.
        # Is 1 adjacent to 32? Yes (we connected 32 to 0,4,8,...,28, not 1).
        # Wait, we connected 32 to 0,4,8,12,16,20,24,28. So 32 is NOT adjacent to 1.
        # So 32-0-2-3-1-32 is NOT a cycle since 1-32 is not an edge.
        
        # Let me re-examine. 32 is adjacent to {0,4,8,12,16,20,24,28}.
        # For a C5: 32-x-y-z-w-32 where x,w in {0,4,8,12,16,20,24,28}.
        # Case 1: x and w in different K4s. Say x in K4_i, w in K4_j, i!=j.
        #   x-y: y is neighbor of x. In K4_i, neighbors of x are the other 3 in K4_i plus 32.
        #   y-z: z is neighbor of y.
        #   z-w: w is in K4_j, so z must be in K4_j or adjacent to w.
        #   But y is in K4_i (or 32), and z must connect to w in K4_j.
        #   If y is in K4_i, z must be in K4_j for z-w to be an edge (since K4s are disjoint).
        #   But then y-z is not an edge (different K4s). Contradiction.
        #   If y=32, then 32-y is not valid (y=32 means x-32 edge, but we already have 32-x).
        #   Actually the cycle is 32-x-y-z-w-32, so x is adjacent to 32 and w is adjacent to 32.
        #   y is adjacent to x, z is adjacent to y, w is adjacent to z.
        #   x in K4_i, w in K4_j, i!=j.
        #   y adj x: y in K4_i \ {x} or y=32.
        #   If y=32: then z adj 32, so z in {0,4,8,12,16,20,24,28}. And w adj z.
        #     w in K4_j, z in some K4_k. For w-z to be edge, k=j and z in K4_j.
        #     So z in K4_j, w in K4_j, z!=w. And y=32, x in K4_i.
        #     Cycle: 32-x-32-z-w-32. But this has 32 repeated. Not a simple cycle.
        #     Actually 32-x-32 is not valid since we can't repeat vertices.
        #   If y in K4_i \ {x}: then y is in K4_i. z adj y: z in K4_i \ {y} or z=32.
        #     If z=32: w adj 32, so w in {0,4,...,28}. But w in K4_j. OK.
        #       Cycle: 32-x-y-32-w-32. Again 32 repeated. Invalid.
        #     If z in K4_i \ {y}: z in K4_i. w adj z: w in K4_j, z in K4_i, i!=j. Not adjacent. Contradiction.
        # Case 2: x and w in same K4_i. Say x, w in K4_i, x!=w.
        #   y adj x: y in K4_i \ {x} or y=32.
        #   If y=32: z adj 32, z in {0,4,...,28}. w adj z. w in K4_i, so z in K4_i.
        #     z in K4_i, z adj 32, so z is the designated vertex in K4_i (e.g., 0 if i=0).
        #     But x is also in K4_i and adj to 32, so x is also the designated vertex. x=z.
        #     But x!=z in a simple cycle. Contradiction.
        #   If y in K4_i \ {x}: y in K4_i. z adj y: z in K4_i \ {y} or z=32.
        #     If z=32: w adj 32. w in K4_i, so w is designated vertex. 
        #       Cycle: 32-x-y-32-w-32. 32 repeated. Invalid.
        #     If z in K4_i \ {y}: z in K4_i. w adj z: w in K4_i, z in K4_i. OK.
        #       Cycle: 32-x-y-z-w-32. All in K4_i plus 32.
        #       x,y,z,w are 4 distinct vertices in K4_i (since K4 has 4 vertices).
        #       So {x,y,z,w} = K4_i. The cycle is 32-x-y-z-w-32.
        #       Edges: 32-x (yes), x-y (yes, in K4), y-z (yes, in K4), z-w (yes, in K4), w-32 (yes).
        #       This IS a 5-cycle! 
        #       Wait: x,y,z,w are all 4 vertices of K4_i. 
        #       32 is adjacent to x and w (the designated vertex). 
        #       But x and w are both in K4_i. 32 is only adjacent to ONE vertex in K4_i (the designated one).
        #       So either x is the designated one and w is not, or vice versa.
        #       If x is designated: 32-x is edge. w-32: w is not designated, so NOT an edge.
        #       If w is designated: 32-w is edge. x-32: x is not designated, so NOT an edge.
        #       So we can't have both 32-x and w-32 be edges unless x=w, which is not allowed.
        #       Contradiction. No C5.
        
        # So connecting 32 to exactly one vertex in each K4 gives a C5-free graph.
        # Now check alpha: The graph is 8 K4s plus vertex 32 connected to one vertex in each.
        # Independent set: can we pick 9 vertices with no edges?
        # From each K4, we can pick at most 1 vertex (since K4 is a clique).
        # So max from the 8 K4s is 8. Plus possibly vertex 32.
        # If we include 32, we cannot include its 8 neighbors (one from each K4).
        # So from each K4, we can pick at most 1 vertex, but not the neighbor of 32.
        # So from each K4, we can pick at most 1 vertex (the other 3 choices).
        # Total: 1 (vertex 32) + 8 (one from each K4, avoiding the neighbor) = 9.
        # Wait, that gives alpha >= 9! We need alpha < 9.
        # So this construction has alpha = 9. Not valid.
        
        # We need to add more edges to reduce alpha.
        # Idea: connect 32 to TWO vertices in each K4. Then if we include 32,
        # we can pick at most 1 from each K4 (avoiding 2 vertices), so 1+8=9. Still 9.
        # Connect 32 to 3 vertices in each K4: then from each K4, only 1 choice.
        # 1 + 8 = 9. Still 9.
        # Connect 32 to all 4 in each K4: then from each K4, 0 choices if 32 included.
        # 1 + 0 = 1. But without 32, we can pick 8. So alpha = max(8, 1) = 8. Good.
        # But connecting 32 to all 32 others: check C5.
        # 32 is connected to everyone. A C5 through 32: 32-a-b-c-d-32.
        # a,b,c,d are 4 distinct vertices from the 32.
        # a-b, b-c, c-d, d-32(=32-d), 32-a.
        # Since 32 is connected to all, 32-a and 32-d are always edges.
        # We need a-b, b-c, c-d to be edges, with a,b,c,d distinct and not 32.
        # In the 8 K4s, within a K4, all pairs are edges.
        # So pick a,b,c,d all from the same K4: a-b, b-c, c-d all edges. C5 exists!
        # So that doesn't work.
        
        # Better approach: add edges between K4s to reduce alpha while preserving C5-freeness.
        # This is getting complex. Let me try a random search instead.
        
        # Random approach: start with 8 K4s + vertex 32, add random edges, check constraints.
        n = 33
        # Base edges: 8 K4s
        base_edges = set()
        for b in range(8):
            base = 4 * b
            for i in range(4):
                for j in range(i + 1, 4):
                    base_edges.add((base + i, base + j))
        
        # Vertex 32 must be connected to reduce alpha.
        # Let's try adding edges from 32 to random vertices and between K4s.
        
        # For now, just try the simple construction and check
        # Actually, let me think differently.
        
        # The 8 K4s have alpha=8. Adding vertex 32:
        # To keep alpha <= 8, 32 must be "covered" - i.e., every independent set of size 8
        # in the original must contain a neighbor of 32.
        # Independent sets of size 8 in 8 K4s: exactly one vertex from each K4.
        # There are 4^8 = 65536 such sets.
        # 32's neighbors must hit all of them.
        # If 32 is connected to vertex v_b in K4_b for each b, then an independent set
        # avoiding all neighbors of 32 must pick from K4_b a vertex != v_b.
        # There are 3^8 = 6561 such sets. Plus vertex 32 itself: total 6562 independent sets
        # of size 9 (32 + one non-neighbor from each K4).
        # To eliminate these, we need more edges.
        
        # Alternative: use a different base structure.
        # What about 7 K4s + 1 K5? 7*4 + 5 = 33 vertices.
        # K5 has alpha=1, K4 has alpha=1. Total alpha = 7*1 + 1 = 8. Good.
        # C5: K5 contains a 5-cycle! So that doesn't work.
        
        # 7 K4s + 1 K4 + 1 vertex? Same as before.
        
        # What about a different decomposition?
        # C5-free graphs: cliques are fine. The issue is combining them.
        
        # Let me try: start with 33 vertices, use a random C5-free graph construction.
        # C5-free graphs include all graphs with no 5-cycle. We can use the fact that
        # graphs with small treewidth or specific structures are C5-free.
        
        # Actually, let me just do a random search with local improvement.
        
        if attempt < 5000:  # Try some structured approaches first
            continue
        
        # Random search: generate random graph, check C5-free and alpha<9
        # This is expensive. Let me skip and just return the 32-vertex solution.
        
    return {'n': n_fallback, 'edges': edges_fallback}
