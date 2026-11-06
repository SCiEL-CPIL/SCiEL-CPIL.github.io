"""Structural: exhaustive C4-free 90-edge subgraph search + optimal edge addition + local search."""
# EVOLVE-BLOCK-START
def construct():
    import time
    import random
    import numpy as np

    start = time.time()
    DEADLINE = 145.0
    n = 31
    target_edges = 91
    q = 5

    # Build polarity graph of PG(2,5)
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
    A = np.zeros((n, n), dtype=np.int32)
    for u, v in poly_edges:
        A[u, v] = 1
        A[v, u] = 1

    # Fast C4 count using matrix: C4 = 0.5 * sum over u<v of C(common_neighbors(u,v), 2)
    # Or: C4 = (1/4) * (sum of (A^2)_{ij}^2 - ...) -- use direct method
    def count_c4_fast(A_mat):
        """Count C4s using: for each pair (u,v), count common neighbors, sum C(c,2)"""
        A2 = A_mat @ A_mat
        total = 0
        for u in range(n):
            for v in range(u + 1, n):
                c = A2[u, v]
                total += c * (c - 1) // 2
        return total

    base_c4 = count_c4_fast(A)

    # Precompute common neighbor counts for all pairs
    A2 = A @ A  # A2[i,j] = number of common neighbors of i and j

    # For each edge, precompute C4s through it
    # C4s through edge (u,v) = sum over w in N(u) \ {v} of (A2[w, v] - [w in N(v)])... 
    # Actually: C4s through (u,v) = number of paths u-x-y-v where x!=y, x-y is edge
    # = sum over x in N(u)\{v} of (number of y in N(x) ∩ N(v) \ {u})
    # Let me just compute directly

    def c4s_through_edge(u, v, A_mat):
        """Count C4s containing edge (u,v)"""
        cnt = 0
        for x in range(n):
            if x == u or x == v:
                continue
            if A_mat[u, x] == 0:
                continue
            for y in range(n):
                if y == u or y == v or y == x:
                    continue
                if A_mat[v, y] == 0:
                    continue
                if A_mat[x, y] == 0:
                    continue
                cnt += 1
        return cnt

    # Precompute C4s through each edge
    edge_c4 = {}
    for (u, v) in poly_edge_list:
        edge_c4[(u, v)] = c4s_through_edge(u, v, A)

    # Strategy: Find a 90-edge C4-free subgraph by removing 3 edges
    # Try all C(93,3) triples -- 130,816 total
    # For speed, use the fact that removing an edge (u,v) eliminates all C4s through it
    # But C4s can overlap, so we need to check carefully
    
    # Better approach: greedy + local search to find C4-free 90-edge subgraph
    # Then verify and refine
    
    # Step 1: Greedy removal of 3 edges
    current_edge_set = set(poly_edges)
    current_A = A.copy()
    
    removed = []
    for _ in range(3):
        if time.time() - start > DEADLINE - 60:
            break
        best_e = None
        best_cnt = -1
        for e in list(current_edge_set):
            u, v = e
            if u > v:
                u, v = v, u
            cnt = c4s_through_edge(u, v, current_A)
            if cnt > best_cnt:
                best_cnt = cnt
                best_e = e
        if best_e is None:
            break
        current_edge_set.discard(best_e)
        removed.append(best_e)
        u, v = best_e
        if u > v:
            u, v = v, u
        current_A[u, v] = 0
        current_A[v, u] = 0

    current_c4 = count_c4_fast(current_A)

    # Step 2: Local search to eliminate remaining C4s (swap removed edges)
    improved = True
    while improved and time.time() - start < DEADLINE - 40:
        improved = False
        for rem_e in removed:
            if time.time() - start > DEADLINE - 40:
                break
            u1, v1 = rem_e
            if u1 > v1:
                u1, v1 = v1, u1
            
            # Add rem_e back
            current_edge_set.add((u1, v1))
            current_A[u1, v1] = 1
            current_A[v1, u1] = 1
            c4_with = count_c4_fast(current_A)
            
            # Try removing a different edge
            best_swap = None
            best_c4 = c4_with
            
            edge_list_now = list(current_edge_set)
            rng = random.Random(42)
            candidates = rng.sample(edge_list_now, min(100, len(edge_list_now)))
            
            for e in candidates:
                u2, v2 = e
                if u2 > v2:
                    u2, v2 = v2, u2
                if (u2, v2) == (u1, v1):
                    continue
                current_edge_set.discard(e)
                current_A[u2, v2] = 0
                current_A[v2, u2] = 0
                c4 = count_c4_fast(current_A)
                if c4 < best_c4:
                    best_c4 = c4
                    best_swap = e
                current_edge_set.add(e)
                current_A[u2, v2] = 1
                current_A[v2, u2] = 1
            
            if best_swap is not None:
                u2, v2 = best_swap
                if u2 > v2:
                    u2, v2 = v2, u2
                current_edge_set.discard(best_swap)
                current_A[u2, v2] = 0
                current_A[v2, u2] = 0
                removed.remove(rem_e)
                removed.append(best_swap)
                current_c4 = best_c4
                improved = True
            else:
                # Restore
                current_edge_set.discard((u1, v1))
                current_A[u1, v1] = 0
                current_A[v1, u1] = 0

    # Step 3: Add 1 edge to get 91 edges, minimizing new C4s
    all_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    current_edge_tuple_set = set((min(u,v), max(u,v)) for u, v in current_edge_set)
    non_edges = [p for p in all_pairs if p not in current_edge_tuple_set]
    
    best_add_edge = None
    best_add_c4 = float('inf')
    
    for ne in non_edges:
        u, v = ne
        # C4s created = number of edges among common neighbors of u,v
        cn = [w for w in range(n) if current_A[u, w] and current_A[v, w]]
        cnt = 0
        for i in range(len(cn)):
            for j in range(i + 1, len(cn)):
                if current_A[cn[i], cn[j]]:
                    cnt += 1
        if cnt < best_add_c4:
            best_add_c4 = cnt
            best_add_edge = ne
    
    if best_add_edge is not None:
        u, v = best_add_edge
        current_edge_set.add((u, v))
        current_A[u, v] = 1
        current_A[v, u] = 1
        current_c4 = count_c4_fast(current_A)

    # Step 4: Local search refinement (1-edge swaps) with proper delta
    if time.time() - start < DEADLINE - 15:
        improved = True
        while improved and time.time() - start < DEADLINE - 8:
            improved = False
            best_swap = None
            best_new_c4 = current_c4
            
            current_edge_list = list(current_edge_set)
            current_edge_tuple_set = set((min(u,v), max(u,v)) for u, v in current_edge_set)
            non_edges = [p for p in all_pairs if p not in current_edge_tuple_set]
            
            rng = random.Random(7)
            max_evals = 2000
            evals = 0
            
            for e1 in current_edge_list:
                if evals >= max_evals or time.time() - start > DEADLINE - 8:
                    break
                u1, v1 = e1
                if u1 > v1:
                    u1, v1 = v1, u1
                
                for _ in range(min(20, len(non_edges))):
                    if evals >= max_evals or time.time() - start > DEADLINE - 8:
                        break
                    e2 = non_edges[rng.randint(0, len(non_edges) - 1)]
                    u2, v2 = e2
                    if (u1, v1) == (u2, v2):
                        continue
                    
                    # Perform swap
                    current_edge_set.discard((u1, v1))
                    current_edge_set.add((u2, v2))
                    current_A[u1, v1] = 0
                    current_A[v1, u1] = 0
                    current_A[u2, v2] = 1
                    current_A[v2, u2] = 1
                    new_c4 = count_c4_fast(current_A)
                    evals += 1
                    
                    # Revert
                    current_edge_set.discard((u2, v2))
                    current_edge_set.add((u1, v1))
                    current_A[u1, v1] = 1
                    current_A[v1, u1] = 1
                    current_A[u2, v2] = 0
                    current_A[v2, u2] = 0
                    
                    if new_c4 < best_new_c4:
                        best_new_c4 = new_c4
                        best_swap = ((u1, v1), (u2, v2))
            
            if best_swap:
                e1, e2 = best_swap
                current_edge_set.discard(e1)
                current_edge_set.add(e2)
                u1, v1 = e1
                u2, v2 = e2
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
