import time
import sys
import random

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 115  # 35 second margin under 150s limit
    
    # 1. Generate vertices of PG(2, 5)
    pts = []
    for x in range(5):
        for y in range(5):
            pts.append((x, y, 1))
    for y in range(5):
        pts.append((1, y, 0))
    pts.append((0, 1, 0))
    
    n = len(pts)
    assert n == 31
    
    # 2. Define Polarity: x1*x2 + y1*y2 + z1*z2 = 0 mod 5
    def is_adjacent(p1, p2):
        x1, y1, z1 = p1
        x2, y2, z2 = p2
        return (x1*x2 + y1*y2 + z1*z2) % 5 == 0
    
    # Build adjacency for core graph (90 edges, C4-free)
    adj = [set() for _ in range(31)]
    edges_core = []
    for i in range(31):
        for j in range(i+1, 31):
            if is_adjacent(pts[i], pts[j]):
                adj[i].add(j)
                adj[j].add(i)
                edges_core.append([i, j])
    
    edge_count = sum(len(s) for s in adj) // 2
    assert edge_count == 90, f"Expected 90 core edges, got {edge_count}"
    
    # 3. Find best edge to add (minimize new C4s)
    best_u = -1
    best_v = -1
    min_paths = float('inf')
    
    for u in range(31):
        if time.time() > deadline - 10:
            break
        for v in range(u+1, 31):
            if v in adj[u]:
                continue
            count = 0
            for a in adj[u]:
                for b in adj[v]:
                    if b in adj[a]:
                        count += 1
            if count < min_paths:
                min_paths = count
                best_u = u
                best_v = v
                if min_paths == 0:
                    break
        if min_paths == 0:
            break
    
    if best_u == -1:
        for u in range(31):
            for v in range(u+1, 31):
                if v not in adj[u]:
                    best_u = u
                    best_v = v
                    break
            if best_u != -1:
                break
    
    # Add the edge
    adj[best_u].add(best_v)
    adj[best_v].add(best_u)
    edges_current = edges_core + [[best_u, best_v]]
    
    # 4. Local search: swap edges to reduce C4 count
    # Helper: count total C4s in current graph
    def count_c4(adj_list):
        total = 0
        n = len(adj_list)
        for i in range(n):
            for j in adj_list[i]:
                if i < j:
                    # Count common neighbors of i and j
                    common = len(adj_list[i] & adj_list[j])
                    total += common
        return total // 2  # Each C4 counted 4 times (once per edge pair)
    
    # Actually, let's use a more direct method:
    # A C4 is a set of 4 vertices {a,b,c,d} with edges ab, bc, cd, da
    # For each pair of vertices, count their common neighbors.
    # Each C4 contributes 2 to the sum (pairs of opposite vertices share 2 common neighbors)
    # So total C4s = sum over all pairs (i,j) of C(common_neighbors(i,j), 2) / 2
    # Wait, that's not right either.
    
    # Correct: For each pair (i,j), let c(i,j) = |N(i) ∩ N(j)|
    # Then total C4s = (1/2) * sum_{i<j} C(c(i,j), 2)
    # Because each C4 has two pairs of opposite vertices, and each such pair has exactly 2 common neighbors (the other two vertices of the C4).
    
    def count_c4_fast(adj_list):
        total = 0
        n = len(adj_list)
        for i in range(n):
            for j in range(i+1, n):
                c = len(adj_list[i] & adj_list[j])
                if c >= 2:
                    total += c * (c - 1) // 2
        return total // 2
    
    current_c4 = count_c4_fast(adj)
    
    # Local search with swaps
    # For a swap: remove edge (u,v), add edge (a,b) where (a,b) is a non-edge
    # Delta in C4 count can be computed incrementally
    
    def delta_c4_remove_add(adj_list, rem_u, rem_v, add_u, add_v):
        """Compute change in C4 count when removing (rem_u,rem_v) and adding (add_u,add_v)"""
        # This is complex; let's just recompute for now and optimize if needed
        # Remove edge
        adj_list[rem_u].discard(rem_v)
        adj_list[rem_v].discard(rem_u)
        
        c4_after_remove = count_c4_fast(adj_list)
        
        # Add edge
        adj_list[add_u].add(add_v)
        adj_list[add_v].add(add_u)
        
        c4_after_add = count_c4_fast(adj_list)
        
        # Restore
        adj_list[add_u].discard(add_v)
        adj_list[add_v].discard(add_u)
        adj_list[rem_u].add(rem_v)
        adj_list[rem_v].add(rem_u)
        
        return c4_after_add - current_c4  # This is wrong; we need delta relative to current state
    
    # Simpler approach: just try swaps and recompute
    best_c4 = current_c4
    best_edges = [e[:] for e in edges_current]
    
    # Build edge set for quick lookup
    edge_set = set()
    for e in edges_current:
        edge_set.add((min(e[0], e[1]), max(e[0], e[1])))
    
    non_edges = []
    for i in range(31):
        for j in range(i+1, 31):
            if (i, j) not in edge_set:
                non_edges.append((i, j))
    
    # Local search
    improved = True
    while improved and time.time() < deadline - 5:
        improved = False
        for rem_idx, rem_edge in enumerate(list(edges_current)):
            if time.time() > deadline - 5:
                break
            ru, rv = rem_edge
            # Try removing this edge
            adj[ru].discard(rv)
            adj[rv].discard(ru)
            
            for add_edge in non_edges:
                au, av = add_edge
                if au == ru or au == rv or av == ru or av == rv:
                    continue
                if (au, av) in edge_set:
                    continue
                # Check if this is a valid swap (not the same edge)
                if (au, av) == (ru, rv) or (au, av) == (rv, ru):
                    continue
                    
                # Add the new edge
                adj[au].add(av)
                adj[av].add(au)
                
                new_c4 = count_c4_fast(adj)
                
                if new_c4 < best_c4:
                    best_c4 = new_c4
                    # Update state
                    edges_current[rem_idx] = [au, av]
                    edge_set.discard((min(ru, rv), max(ru, rv)))
                    edge_set.add((min(au, av), max(au, av)))
                    # Update non_edges
                    non_edges = [(i, j) for i in range(31) for j in range(i+1, 31) if (i, j) not in edge_set]
                    improved = True
                    break
                
                # Undo add
                adj[au].discard(av)
                adj[av].discard(au)
            
            # Undo remove
            adj[ru].add(rv)
            adj[rv].add(ru)
            
            if improved:
                break
    
    # Reconstruct final edge list from best state
    # Actually, let's track the best state properly
    # The above loop modifies edges_current in place when improvement is found
    # So edges_current should hold the best state
    
    final_edges = [e[:] for e in edges_current]
    
    # Verify
    assert len(final_edges) == 91, f"Expected 91 edges, got {len(final_edges)}"
    assert len(set(tuple(sorted(e)) for e in final_edges)) == 91
    
    return final_edges
