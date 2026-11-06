"""Combined: C4-free base + minimal edge addition + local search refinement."""
# EVOLVE-BLOCK-START
def construct():
    import time
    import random
    import numpy as np

    start = time.time()
    DEADLINE = 145.0
    n = 31
    target_edges = 91

    # ---- Build polarity graph of PG(2,5): 31 vertices, 93 edges ----
    q = 5
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
    m = len(poly_edge_list)  # should be 93

    # ---- Fast C4 counting ----
    def count_c4(adj_sets):
        total = 0
        for u in range(n):
            nu = adj_sets[u]
            for v in range(u + 1, n):
                c = len(nu & adj_sets[v])
                total += c * (c - 1) // 2
        return total

    def build_adj_sets(edge_set):
        adj = [set() for _ in range(n)]
        for u, v in edge_set:
            adj[u].add(v)
            adj[v].add(u)
        return adj

    # ---- Step 1: Find C4-free 90-edge subgraph by removing 3 edges ----
    # The polarity graph is (31,6,2,1) srg with 93 edges and 45 C4s.
    # We need to remove 3 edges to eliminate all C4s.
    # Each C4 uses 4 edges. Removing 3 well-chosen edges can hit many C4s.
    
    # Build initial adjacency
    adj0 = build_adj_sets(poly_edges)
    base_c4 = count_c4(adj0)  # should be 45

    # For each edge, compute how many C4s it's in
    def c4s_through_edge(u, v, adj_sets):
        cn = adj_sets[u] & adj_sets[v]
        cn_list = list(cn)
        cnt = 0
        for i in range(len(cn_list)):
            a = cn_list[i]
            for j in range(i + 1, len(cn_list)):
                b = cn_list[j]
                if b in adj_sets[a]:
                    cnt += 1
        return cnt

    edge_c4_count = {}
    for (u, v) in poly_edge_list:
        edge_c4_count[(u, v)] = c4s_through_edge(u, v, adj0)

    # Greedy: remove edges that are in the most C4s
    # Try all C(93,3) = 130,816 triples? Too many for full C4 count.
    # Instead: greedy removal + local search
    
    # Greedy: iteratively remove edge with max C4 count
    current_edge_set = set(poly_edges)
    current_adj = build_adj_sets(current_edge_set)
    
    # Remove 3 edges greedily
    removed_edges = []
    for _ in range(3):
        if time.time() - start > DEADLINE - 30:
            break
        best_edge = None
        best_count = -1
        for e in list(current_edge_set):
            u, v = e
            if u > v:
                u, v = v, u
            # Count C4s through this edge in current graph
            cnt = c4s_through_edge(u, v, current_adj)
            if cnt > best_count:
                best_count = cnt
                best_edge = e
        if best_edge is None:
            break
        current_edge_set.discard(best_edge)
        removed_edges.append(best_edge)
        current_adj = build_adj_sets(current_edge_set)
    
    current_c4 = count_c4(current_adj)
    
    # Local search to find C4-free 90-edge subgraph
    # Try swapping removed edges with other edges to reduce C4
    improved = True
    while improved and time.time() - start < DEADLINE - 20:
        improved = False
        # Try: for each removed edge, try re-adding it and removing a different edge
        for rem_e in removed_edges:
            if time.time() - start > DEADLINE - 20:
                break
            # Add rem_e back
            u1, v1 = rem_e
            if u1 > v1:
                u1, v1 = v1, u1
            current_edge_set.add((u1, v1))
            current_adj = build_adj_sets(current_edge_set)
            
            # Now try removing a different edge to get back to 90 edges
            best_swap = None
            best_c4 = count_c4(current_adj)
            
            # Sample edges to remove
            edge_list_now = list(current_edge_set)
            rng = random.Random(42)
            candidates = rng.sample(edge_list_now, min(200, len(edge_list_now)))
            
            for e in candidates:
                u2, v2 = e
                if u2 > v2:
                    u2, v2 = v2, u2
                if (u2, v2) == (u1, v1):
                    continue
                # Remove e
                current_edge_set.discard(e)
                current_adj = build_adj_sets(current_edge_set)
                c4 = count_c4(current_adj)
                if c4 < best_c4:
                    best_c4 = c4
                    best_swap = e
                # Restore
                current_edge_set.add(e)
                current_adj = build_adj_sets(current_edge_set)
            
            if best_swap is not None:
                # Apply: remove best_swap, keep rem_e
                current_edge_set.discard(best_swap)
                current_adj = build_adj_sets(current_edge_set)
                removed_edges.remove(rem_e)
                removed_edges.append(best_swap)
                current_c4 = best_c4
                improved = True
            else:
                # Restore: remove rem_e again
                current_edge_set.discard((u1, v1))
                current_adj = build_adj_sets(current_edge_set)
    
    # ---- Step 2: Add 1 edge to get 91 edges, minimizing new C4s ----
    # Non-edges in current 90-edge graph
    all_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    current_edge_tuple_set = set((min(u,v), max(u,v)) for u, v in current_edge_set)
    non_edges = [p for p in all_pairs if p not in current_edge_tuple_set]
    
    # For each non-edge, compute C4s it would create
    best_add_edge = None
    best_add_c4 = float('inf')
    
    for ne in non_edges:
        u, v = ne
        # C4s created by adding edge (u,v) = number of edges among common neighbors of u,v
        cn = current_adj[u] & current_adj[v]
        cn_list = list(cn)
        cnt = 0
        for i in range(len(cn_list)):
            a = cn_list[i]
            for j in range(i + 1, len(cn_list)):
                b = cn_list[j]
                if b in current_adj[a]:
                    cnt += 1
        if cnt < best_add_c4:
            best_add_c4 = cnt
            best_add_edge = ne
    
    if best_add_edge is not None:
        u, v = best_add_edge
        current_edge_set.add((u, v))
        current_adj = build_adj_sets(current_edge_set)
        current_c4 = count_c4(current_adj)
    
    # ---- Step 3: Local search refinement (1-edge swaps) ----
    if time.time() - start < DEADLINE - 10:
        improved = True
        while improved and time.time() - start < DEADLINE - 5:
            improved = False
            best_swap = None
            best_new_c4 = current_c4
            
            current_edge_list = list(current_edge_set)
            current_edge_tuple_set = set((min(u,v), max(u,v)) for u, v in current_edge_set)
            non_edges = [p for p in all_pairs if p not in current_edge_tuple_set]
            
            rng = random.Random(7)
            max_evals = 3000
            evals = 0
            
            for e1 in current_edge_list:
                if evals >= max_evals or time.time() - start > DEADLINE - 5:
                    break
                u1, v1 = e1
                if u1 > v1:
                    u1, v1 = v1, u1
                
                # Try adding random non-edges
                for _ in range(min(30, len(non_edges))):
                    if evals >= max_evals or time.time() - start > DEADLINE - 5:
                        break
                    e2 = non_edges[rng.randint(0, len(non_edges) - 1)]
                    u2, v2 = e2
                    if (u1, v1) == (u2, v2):
                        continue
                    
                    # Perform swap
                    current_edge_set.discard((u1, v1))
                    current_edge_set.add((u2, v2))
                    current_adj = build_adj_sets(current_edge_set)
                    new_c4 = count_c4(current_adj)
                    evals += 1
                    
                    # Revert
                    current_edge_set.discard((u2, v2))
                    current_edge_set.add((u1, v1))
                    current_adj = build_adj_sets(current_edge_set)
                    
                    if new_c4 < best_new_c4:
                        best_new_c4 = new_c4
                        best_swap = ((u1, v1), (u2, v2))
            
            if best_swap:
                e1, e2 = best_swap
                current_edge_set.discard(e1)
                current_edge_set.add(e2)
                current_adj = build_adj_sets(current_edge_set)
                current_c4 = best_new_c4
                improved = True
    
    # ---- Step 4: 2-edge swaps if time permits ----
    if time.time() - start < DEADLINE - 10:
        improved2 = True
        while improved2 and time.time() - start < DEADLINE - 5:
            improved2 = False
            best_swap2 = None
            best_new_c4_2 = current_c4
            
            current_edge_list = list(current_edge_set)
            current_edge_tuple_set = set((min(u,v), max(u,v)) for u, v in current_edge_set)
            non_edges = [p for p in all_pairs if p not in current_edge_tuple_set]
            
            rng2 = random.Random(123)
            max_evals2 = 2000
            evals2 = 0
            
            for _ in range(max_evals2):
                if time.time() - start > DEADLINE - 5:
                    break
                if len(current_edge_list) < 2 or len(non_edges) < 2:
                    break
                idx1, idx2 = rng2.sample(range(len(current_edge_list)), 2)
                e1 = current_edge_list[idx1]
                e2 = current_edge_list[idx2]
                idx3, idx4 = rng2.sample(range(len(non_edges)), 2)
                e3 = non_edges[idx3]
                e4 = non_edges[idx4]
                
                u1, v1 = e1
                u2, v2 = e2
                u3, v3 = e3
                u4, v4 = e4
                
                # Check no conflicts
                if (u1, v1) == (u3, v3) or (u1, v1) == (u4, v4) or \
                   (u2, v2) == (u3, v3) or (u2, v2) == (u4, v4):
                    continue
                if e3 == e4:
                    continue
                
                # Perform 2-edge swap
                current_edge_set.discard(e1)
                current_edge_set.discard(e2)
                current_edge_set.add(e3)
                current_edge_set.add(e4)
                current_adj = build_adj_sets(current_edge_set)
                new_c4 = count_c4(current_adj)
                evals2 += 1
                
                # Revert
                current_edge_set.discard(e3)
                current_edge_set.discard(e4)
                current_edge_set.add(e1)
                current_edge_set.add(e2)
                current_adj = build_adj_sets(current_edge_set)
                
                if new_c4 < best_new_c4_2:
                    best_new_c4_2 = new_c4
                    best_swap2 = (e1, e2, e3, e4)
            
            if best_swap2:
                e1, e2, e3, e4 = best_swap2
                current_edge_set.discard(e1)
                current_edge_set.discard(e2)
                current_edge_set.add(e3)
                current_edge_set.add(e4)
                current_adj = build_adj_sets(current_edge_set)
                current_c4 = best_new_c4_2
                improved2 = True
    
    # ---- Return exactly 91 edges ----
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
