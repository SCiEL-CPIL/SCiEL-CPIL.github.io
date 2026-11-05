"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time

def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin before 150s kill
    n = 31
    target_edges = 91

    def build_polarity_graph():
        """Construct a polarity graph on PG(2,5) - C4-free with 90 edges."""
        q = 5
        points = []
        seen = set()
        for x0 in range(q):
            for x1 in range(q):
                for x2 in range(q):
                    if x0 == 0 and x1 == 0 and x2 == 0:
                        continue
                    if x0 != 0:
                        inv = pow(x0, q-2, q)
                        canon = (1, x1 * inv % q, x2 * inv % q)
                    elif x1 != 0:
                        inv = pow(x1, q-2, q)
                        canon = (0, 1, x2 * inv % q)
                    else:
                        canon = (0, 0, 1)
                    if canon not in seen:
                        seen.add(canon)
                        points.append(canon)
        
        assert len(points) == 31, f"Expected 31 points, got {len(points)}"
        
        def are_conjugate(p, r):
            return (p[0]*r[2] + p[1]*r[1] + p[2]*r[0]) % q == 0
        
        adj = [set() for _ in range(31)]
        edges = []
        for i in range(31):
            for j in range(i+1, 31):
                if are_conjugate(points[i], points[j]):
                    adj[i].add(j)
                    adj[j].add(i)
                    edges.append([i, j])
        
        return edges, adj

    def count_c4(adj_sets):
        """Count 4-cycles efficiently."""
        total = 0
        for u in range(n):
            for v in range(u + 1, n):
                if v not in adj_sets[u]:
                    common = len(adj_sets[u] & adj_sets[v])
                    if common >= 2:
                        total += common * (common - 1) // 2
        return total // 2

    # Step 1: Build the polarity graph (C4-free, 90 edges)
    core_edges, core_adj = build_polarity_graph()
    
    if len(core_edges) != 90:
        # Fallback: greedy C4-free construction
        core_edges = []
        core_adj = [set() for _ in range(n)]
        all_pairs = list(range(n*(n-1)//2))
        random.shuffle(all_pairs)
        for idx in all_pairs:
            if len(core_edges) >= 90:
                break
            u = idx // (n - 1)
            v = (idx % (n - 1)) + u + 1
            if u >= v:
                continue
            common = core_adj[u] & core_adj[v]
            if len(common) < 2:
                core_edges.append([u, v])
                core_adj[u].add(v)
                core_adj[v].add(u)
    
    # Step 2: Add the 91st edge to minimize total C4
    edge_set = set()
    for e in core_edges:
        edge_set.add(tuple(sorted(e)))
    
    best_add_c4 = float('inf')
    best_add_edge = None
    for i in range(n):
        for j in range(i+1, n):
            if (i, j) not in edge_set:
                common = core_adj[i] & core_adj[j]
                new_c4s = len(common) * (len(common) - 1) // 2
                if new_c4s < best_add_c4:
                    best_add_c4 = new_c4s
                    best_add_edge = (i, j)
    
    if best_add_edge is not None:
        core_edges.append(list(best_add_edge))
        core_adj[best_add_edge[0]].add(best_add_edge[1])
        core_adj[best_add_edge[1]].add(best_add_edge[0])
        edge_set.add(best_add_edge)
    
    current_c4 = count_c4(core_adj)
    
    # Step 3: Aggressive local search with multiple passes
    if time.time() < deadline - 5:
        # Build adjacency from core_edges
        adj = [set() for _ in range(n)]
        for e in core_edges:
            adj[e[0]].add(e[1])
            adj[e[1]].add(e[0])
        
        edge_set = set(tuple(sorted(e)) for e in core_edges)
        
        # Precompute non-edges
        all_non_edges = []
        for i in range(n):
            for j in range(i+1, n):
                if (i,j) not in edge_set:
                    all_non_edges.append((i,j))
        
        # Multi-pass local search: try all single-edge swaps, repeat until no improvement
        max_passes = 50
        for pass_num in range(max_passes):
            if time.time() > deadline - 5:
                break
            
            improved = False
            edge_list = [list(e) for e in edge_set]
            random.shuffle(edge_list)
            
            for e_idx in range(len(edge_list)):
                if time.time() > deadline - 5:
                    break
                u, v = edge_list[e_idx]
                u, v = sorted([u, v])
                
                # Delta from removing (u,v): destroys C4s where u,v adjacent
                common_uv = adj[u] & adj[v]
                delta_remove = -len(common_uv) * (len(common_uv) - 1) // 2
                
                # Temporarily remove edge (u,v)
                adj[u].discard(v)
                adj[v].discard(u)
                edge_set.discard((u, v))
                
                best_delta = 0
                best_new_edge = None
                
                for (a, b) in all_non_edges:
                    if (a, b) == (u, v):
                        continue
                    # Delta from adding (a,b) to graph without (u,v)
                    common_ab = adj[a] & adj[b]
                    delta_add = len(common_ab) * (len(common_ab) - 1) // 2
                    total_delta = delta_remove + delta_add
                    
                    if total_delta < best_delta:
                        best_delta = total_delta
                        best_new_edge = (a, b)
                
                # Restore (u,v)
                adj[u].add(v)
                adj[v].add(u)
                edge_set.add((u, v))
                
                if best_new_edge is not None and best_delta < 0:
                    # Perform the swap
                    a, b = best_new_edge
                    adj[u].discard(v)
                    adj[v].discard(u)
                    edge_set.discard((u, v))
                    adj[a].add(b)
                    adj[b].add(a)
                    edge_set.add((a, b))
                    current_c4 += best_delta
                    improved = True
                    # Update non-edges list
                    all_non_edges.remove((a, b))
                    all_non_edges.append((u, v))
                    break
            
            if not improved:
                break
        
        # Step 4: 2-opt moves (swap two edges for two non-edges)
        if time.time() < deadline - 10:
            edge_list = list(edge_set)
            for i in range(len(edge_list)):
                if time.time() > deadline - 10:
                    break
                e1 = edge_list[i]
                u1, v1 = sorted(e1)
                
                for j in range(i+1, len(edge_list)):
                    if time.time() > deadline - 10:
                        break
                    e2 = edge_list[j]
                    u2, v2 = sorted(e2)
                    
                    # Temporarily remove both edges
                    adj[u1].discard(v1)
                    adj[v1].discard(u1)
                    adj[u2].discard(v2)
                    adj[v2].discard(u2)
                    edge_set.discard(tuple(e1))
                    edge_set.discard(tuple(e2))
                    
                    delta_remove1 = -len((adj[u1] & adj[v1])) * (len(adj[u1] & adj[v1]) - 1) // 2
                    delta_remove2 = -len((adj[u2] & adj[v2])) * (len(adj[u2] & adj[v2]) - 1) // 2
                    # Account for shared common neighbors
                    shared_common = (adj[u1] & adj[v1]) & (adj[u2] & adj[v2])
                    # This is approximate; for exact, we'd need more careful accounting
                    # For now, use a simpler approach: just check if total improves
                    
                    best_delta = 0
                    best_new_edges = None
                    
                    # Try adding pairs of non-edges
                    non_edges_copy = list(all_non_edges)
                    for a_idx in range(len(non_edges_copy)):
                        if time.time() > deadline - 10:
                            break
                        a1, b1 = non_edges_copy[a_idx]
                        if (a1, b1) == (u1, v1) or (a1, b1) == (u2, v2):
                            continue
                        
                        for b_idx in range(a_idx+1, len(non_edges_copy)):
                            if time.time() > deadline - 15:
                                break
                            a2, b2 = non_edges_copy[b_idx]
                            if (a2, b2) == (u1, v1) or (a2, b2) == (u2, v2):
                                continue
                            if (a2, b2) == (a1, b1):
                                continue
                            
                            # Compute delta for adding both
                            common_ab1 = adj[a1] & adj[b1]
                            common_ab2 = adj[a2] & adj[b2]
                            delta_add1 = len(common_ab1) * (len(common_ab1) - 1) // 2
                            delta_add2 = len(common_ab2) * (len(common_ab2) - 1) // 2
                            total_delta = delta_remove1 + delta_remove2 + delta_add1 + delta_add2
                            
                            if total_delta < best_delta:
                                best_delta = total_delta
                                best_new_edges = ((a1, b1), (a2, b2))
                    
                    # Restore both edges
                    adj[u1].add(v1)
                    adj[v1].add(u1)
                    adj[u2].add(v2)
                    adj[v2].add(u2)
                    edge_set.add(tuple(e1))
                    edge_set.add(tuple(e2))
                    
                    if best_new_edges is not None and best_delta < 0:
                        (a1, b1), (a2, b2) = best_new_edges
                        adj[u1].discard(v1)
                        adj[v1].discard(u1)
                        adj[u2].discard(v2)
                        adj[v2].discard(u2)
                        edge_set.discard(tuple(e1))
                        edge_set.discard(tuple(e2))
                        adj[a1].add(b1)
                        adj[b1].add(a1)
                        adj[a2].add(b2)
                        adj[b2].add(a2)
                        edge_set.add((a1, b1))
                        edge_set.add((a2, b2))
                        current_c4 += best_delta
                        # Update non-edges
                        all_non_edges.remove((a1, b1))
                        all_non_edges.remove((a2, b2))
                        all_non_edges.append((u1, v1))
                        all_non_edges.append((u2, v2))
                        break
        
        # Final verification
        current_c4 = count_c4(adj)
        core_edges = [list(sorted(e)) for e in edge_set]
    
    # Ensure exactly 91 distinct edges
    edge_set = set()
    unique_edges = []
    for e in core_edges:
        key = tuple(sorted(e))
        if key not in edge_set:
            edge_set.add(key)
            unique_edges.append(list(e))
    
    if len(unique_edges) > target_edges:
        unique_edges = unique_edges[:target_edges]
    
    if len(unique_edges) < target_edges:
        adj_temp = [set() for _ in range(n)]
        for e in unique_edges:
            adj_temp[e[0]].add(e[1])
            adj_temp[e[1]].add(e[0])
        for i in range(n):
            for j in range(i+1, n):
                if len(unique_edges) >= target_edges:
                    break
                if (i, j) not in edge_set:
                    unique_edges.append([i, j])
                    edge_set.add((i, j))
            if len(unique_edges) >= target_edges:
                break
    
    return unique_edges[:target_edges]
# EVOLVE-BLOCK-END
