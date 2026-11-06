"""Structural: Exhaustive C4-free subgraph search + optimal edge addition + efficient local search."""
# EVOLVE-BLOCK-START
def construct():
    import time
    import random
    import numpy as np
    from itertools import combinations

    start = time.time()
    DEADLINE = 148.0
    n = 31
    target_edges = 91
    q = 5

    # 1. Build polarity graph of PG(2,5)
    points = []
    for x in range(q):
        for y in range(q):
            points.append((x, y, 1))
    for x in range(q):
        points.append((x, 1, 0))
    points.append((1, 0, 0))

    poly_edges = set()
    for i in range(n):
        for j in range(i + 1, n):
            p1, p2 = points[i], points[j]
            if (p1[0]*p2[0] + p1[1]*p2[1] - p1[2]*p2[2]) % q == 0:
                poly_edges.add((i, j))

    poly_edge_list = list(poly_edges)
    m = len(poly_edge_list)  # 93

    # Build adjacency matrix
    A = np.zeros((n, n), dtype=np.int8)
    for u, v in poly_edges:
        A[u, v] = 1
        A[v, u] = 1

    # Helper to count C4s
    def count_c4_fast(A_mat):
        A2 = A_mat @ A_mat
        total = 0
        # Vectorized sum of C(c, 2) for all pairs u < v
        # c = A2[u, v]
        # We can use np.triu_indices to get upper triangle
        iu, ju = np.triu_indices(n, 1)
        c_vals = A2[iu, ju]
        total = np.sum(c_vals * (c_vals - 1) // 2)
        return int(total)

    # Helper to calculate delta C4 for adding an edge (u, v) to graph with adjacency A
    # New C4s = number of edges in the subgraph induced by common neighbors of u and v
    def delta_add_edge(u, v, A_mat):
        # Common neighbors
        cn_mask = (A_mat[u, :] == 1) & (A_mat[v, :] == 1)
        cn_indices = np.where(cn_mask)[0]
        if len(cn_indices) < 2:
            return 0
        # Count edges among common neighbors
        sub_A = A_mat[np.ix_(cn_indices, cn_indices)]
        # Number of edges is (sum of sub_A) / 2
        return int(np.sum(sub_A) // 2)

    # Helper to calculate delta C4 for removing an edge (u, v) from graph with adjacency A
    # Removed C4s = number of paths u-x-y-v where x-y is an edge and x,y are distinct from u,v
    # This is equivalent to counting C4s containing (u,v)
    def delta_remove_edge(u, v, A_mat):
        # C4s containing (u,v) correspond to triangles in the graph formed by N(u) \ {v} and N(v) \ {u}
        # More simply: for each x in N(u)\{v}, count y in N(v)\{u} such that (x,y) is an edge.
        # But we must ensure x != y.
        # Let's use the formula:
        # C4s through (u,v) = sum_{x in N(u)\{v}} ( |N(x) ∩ N(v)| - [u in N(v)] )
        # Wait, if x is neighbor of u, and y is neighbor of v, and (x,y) is edge, we have u-x-y-v.
        # If x=v, not allowed. If y=u, not allowed.
        # So we sum over x in N(u) \ {v} the number of y in N(v) \ {u} such that A[x,y]=1.
        
        cnt = 0
        for x in range(n):
            if x == u or x == v: continue
            if A_mat[u, x] == 0: continue
            # Count y in N(v) \ {u} such that A[x,y]=1
            # y must be neighbor of v
            # y must be neighbor of x
            # y must not be u
            # y must not be v (implied if A[x,y]=1 and x!=v? No, x can be v? No x!=v. 
            # If y=v, A[x,v]=1. But y must be neighbor of v, so A[v,v]=0. So y!=v automatically if A[v,y]=1? No.
            # A[v,y]=1 implies y!=v.
            # So we just need A[x,y]=1 and A[v,y]=1 and y!=u.
            
            # Vectorized:
            # mask_y = (A_mat[v, :] == 1) & (A_mat[x, :] == 1) & (np.arange(n) != u)
            # cnt += sum(mask_y)
            
            # Optimization: Precomputed? No, let's just do it. n=31 is small.
            for y in range(n):
                if y == u or y == v: continue
                if A_mat[v, y] == 1 and A_mat[x, y] == 1:
                    cnt += 1
        return cnt

    # Step 1: Find a C4-free 90-edge subgraph
    # The polarity graph has 93 edges. We need to remove 3.
    # Let's try to find 3 edges whose removal eliminates all C4s.
    # First, count total C4s in PG(2,5)
    total_c4 = count_c4_fast(A)
    
    # If total_c4 is 0, we are done with step 1 (but PG(2,5) is C4-free? 
    # Wait, PG(2,q) polarity graph is C4-free for q odd? 
    # Actually, the polarity graph of PG(2,5) has 31 vertices and 93 edges.
    # It is known to be C4-free? 
    # Let's check: ex(31, C4) = 90. So a 93-edge graph MUST have C4s.
    # So total_c4 > 0.
    
    # We need to remove 3 edges to make it C4-free.
    # This is a hitting set problem for C4s.
    # Since the graph is small, we can iterate through combinations of 3 edges.
    # There are C(93, 3) ~ 130k combinations.
    # For each, we check if the remaining graph is C4-free.
    # Checking C4-freeness takes O(n^3) or similar. 130k * 31^3 is too slow.
    
    # Alternative: 
    # 1. Identify all C4s in the original graph.
    # 2. Find a set of 3 edges that hits all C4s.
    
    # Let's list all C4s.
    # A C4 is a set of 4 vertices {a,b,c,d} with edges (a,b), (b,c), (c,d), (d,a).
    # We can find them by looking at pairs of vertices with common neighbors.
    
    c4s = []
    A2 = A @ A
    for u in range(n):
        for v in range(u + 1, n):
            c = A2[u, v]
            if c >= 2:
                # Find common neighbors
                cn = np.where((A[u, :] == 1) & (A[v, :] == 1))[0]
                # Any pair of common neighbors (x, y) forms a C4 u-x-v-y-u
                for i in range(len(cn)):
                    for j in range(i + 1, len(cn)):
                        x, y = cn[i], cn[j]
                        # The C4 is (u, x, v, y). Edges: (u,x), (x,v), (v,y), (y,u)
                        # We store the set of edges
                        e1 = (min(u, x), max(u, x))
                        e2 = (min(x, v), max(x, v))
                        e3 = (min(v, y), max(v, y))
                        e4 = (min(y, u), max(y, u))
                        c4_edges = frozenset([e1, e2, e3, e4])
                        c4s.append(c4_edges)
    
    # Now we have a list of C4s (as frozensets of edges).
    # We want to remove 3 edges such that every C4 contains at least one of the removed edges.
    # This is a Set Cover / Hitting Set problem.
    # Since we only remove 3, we can try all pairs of edges from the first C4?
    # Or just iterate through all triples of edges from the graph.
    
    # Optimization: Only consider edges that are in at least one C4.
    edges_in_c4 = set()
    for c4 in c4s:
        edges_in_c4.update(c4)
    
    candidate_edges = list(edges_in_c4)
    
    found_c4_free = False
    best_removed = None
    
    # Iterate through combinations of 3 candidate edges
    # If the number of candidate edges is large, this might be slow.
    # But typically, not all 93 edges are in C4s.
    # Let's cap the search time.
    search_start = time.time()
    
    for e1, e2, e3 in combinations(candidate_edges, 3):
        if time.time() - search_start > 5.0:
            break
            
        # Check if {e1, e2, e3} hits all C4s
        hit_all = True
        for c4 in c4s:
            if not (e1 in c4 or e2 in c4 or e3 in c4):
                hit_all = False
                break
        if hit_all:
            best_removed = [e1, e2, e3]
            found_c4_free = True
            break
            
    # If we didn't find a 3-edge hitting set, we might need to accept some C4s
    # or use a different strategy. But given ex(31, C4)=90, it's likely possible.
    
    current_edge_set = set(poly_edges)
    current_A = A.copy()
    
    if found_c4_free:
        for e in best_removed:
            u, v = e
            current_edge_set.discard(e)
            current_A[u, v] = 0
            current_A[v, u] = 0
        current_c4 = 0
    else:
        # Fallback: Greedy removal
        # Just remove the 3 edges with highest C4 count
        edge_c4_counts = {}
        for e in poly_edge_list:
            u, v = e
            edge_c4_counts[e] = delta_remove_edge(u, v, A)
        
        # Sort by count descending
        sorted_edges = sorted(edge_c4_counts.items(), key=lambda x: x[1], reverse=True)
        for e, cnt in sorted_edges[:3]:
            u, v = e
            current_edge_set.discard(e)
            current_A[u, v] = 0
            current_A[v, u] = 0
        
        current_c4 = count_c4_fast(current_A)

    # Step 2: Add 1 edge to get 91 edges, minimizing new C4s
    all_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    current_edge_tuple_set = set((min(u,v), max(u,v)) for u, v in current_edge_set)
    non_edges = [p for p in all_pairs if p not in current_edge_tuple_set]
    
    best_add_edge = None
    best_add_c4 = float('inf')
    
    for ne in non_edges:
        u, v = ne
        cnt = delta_add_edge(u, v, current_A)
        if cnt < best_add_c4:
            best_add_c4 = cnt
            best_add_edge = ne
    
    if best_add_edge is not None:
        u, v = best_add_edge
        current_edge_set.add((u, v))
        current_A[u, v] = 1
        current_A[v, u] = 1
        current_c4 = count_c4_fast(current_A)

    # Step 3: Local Search Refinement
    # Swap an existing edge with a non-edge to reduce C4 count
    # Delta = C4s(removing e1) - C4s(adding e2) + interaction?
    # It's easier to just calculate the new C4 count if the delta is complex.
    # But calculating full C4 count is O(n^3) ~ 30k ops.
    # We can do many iterations.
    
    improved = True
    while improved and time.time() - start < DEADLINE - 5:
        improved = False
        best_swap = None
        best_new_c4 = current_c4
        
        current_edge_list = list(current_edge_set)
        current_edge_tuple_set = set((min(u,v), max(u,v)) for u, v in current_edge_set)
        non_edges = [p for p in all_pairs if p not in current_edge_tuple_set]
        
        # Randomize order
        rng = random.Random(time.time_ns())
        rng.shuffle(current_edge_list)
        
        # Limit evaluations per pass to avoid timeout
        max_evals = 500
        evals = 0
        
        for e1 in current_edge_list:
            if evals >= max_evals or time.time() - start > DEADLINE - 5:
                break
                
            u1, v1 = e1
            # Try a few random non-edges
            for _ in range(10):
                if evals >= max_evals or time.time() - start > DEADLINE - 5:
                    break
                if not non_edges: break
                e2 = rng.choice(non_edges)
                u2, v2 = e2
                
                # Calculate delta
                # Remove e1, Add e2
                # New C4s = Old C4s - C4s(e1) + C4s(e2) + Interaction
                # Interaction is tricky. Let's just compute full count if we want to be safe,
                # or use a fast approximate delta.
                
                # Let's try a fast delta:
                # C4s lost by removing e1: delta_remove_edge(u1, v1, current_A)
                # C4s gained by adding e2: delta_add_edge(u2, v2, current_A)
                # BUT: The removal of e1 changes the graph, so delta_add_edge for e2 might change.
                # And the C4s lost by e1 might include e2? No, e2 is not in graph.
                # However, C4s containing e1 might be affected by the presence of e2? No, e2 is added.
                # The interaction is: C4s that contain BOTH e1 and e2? No, e2 is not there yet.
                # C4s that contain e1 and are destroyed by removing e1.
                # C4s that contain e2 and are created by adding e2.
                # Are there C4s that are destroyed by removing e1 AND created by adding e2?
                # If a C4 contains e1 and e2, it doesn't exist before (no e2) and doesn't exist after (no e1).
                # So they cancel out?
                # Let's assume independence for a quick check, then verify.
                
                c4_lost = delta_remove_edge(u1, v1, current_A)
                c4_gained = delta_add_edge(u2, v2, current_A)
                estimated_delta = c4_gained - c4_lost
                
                if estimated_delta < 0:
                    # Verify with full count
                    current_edge_set.discard(e1)
                    current_edge_set.add(e2)
                    current_A[u1, v1] = 0
                    current_A[v1, u1] = 0
                    current_A[u2, v2] = 1
                    current_A[v2, u2] = 1
                    
                    new_c4 = count_c4_fast(current_A)
                    evals += 1
                    
                    if new_c4 < best_new_c4:
                        best_new_c4 = new_c4
                        best_swap = (e1, e2)
                        # Keep the change for now, revert if we find better or if we finish
                        # Actually, let's revert to keep state consistent for other checks
                    current_edge_set.discard(e2)
                    current_edge_set.add(e1)
                    current_A[u1, v1] = 1
                    current_A[v1, u1] = 1
                    current_A[u2, v2] = 0
                    current_A[v2, u2] = 0
        
        if best_swap:
            e1, e2 = best_swap
            u1, v1 = e1
            u2, v2 = e2
            current_edge_set.discard(e1)
            current_edge_set.add(e2)
            current_A[u1, v1] = 0
            current_A[v1, u1] = 0
            current_A[u2, v2] = 1
            current_A[v2, u2] = 1
            current_c4 = best_new_c4
            improved = True

    # Return exactly 91 edges
    final_edges = list(current_edge_set)
    if len(final_edges) != target_edges:
        if len(final_edges) > target_edges:
            final_edges = final_edges[:target_edges]
        elif len(final_edges) < target_edges:
            current_edge_tuple_set = set((min(u,v), max(u,v)) for u, v in final_edges)
            for ne in all_pairs:
                if len(final_edges) >= target_edges:
                    break
                if ne not in current_edge_tuple_set:
                    final_edges.append(ne)

    return [[u, v] for u, v in final_edges]
# EVOLVE-BLOCK-END
