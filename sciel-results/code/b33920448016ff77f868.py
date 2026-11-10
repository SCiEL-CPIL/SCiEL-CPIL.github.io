import time
import numpy as np
import random

def construct():
    start_time = time.time()
    deadline = start_time + 145  # safety margin before 150s limit
    
    q = 5
    # Build PG(2, q) points
    points = []
    for x in range(q):
        for y in range(q):
            points.append((1, x, y))
    for z in range(q):
        points.append((0, 1, z))
    points.append((0, 0, 1))
    
    n = len(points)  # should be 31
    
    def dot(p, r):
        return (p[0]*r[0] + p[1]*r[1] + p[2]*r[2]) % q
    
    # Build polarity graph: edge between i,j if dot(points[i], points[j]) == 0 and i != j
    edges = []
    for i in range(n):
        for j in range(i+1, n):
            if dot(points[i], points[j]) == 0:
                edges.append([i, j])
    
    # Convert to adjacency matrix (numpy for speed)
    A = np.zeros((n, n), dtype=np.int64)
    for e in edges:
        A[e[0], e[1]] = 1
        A[e[1], e[0]] = 1
    
    def count_4cycles_fast(A_mat):
        """Faster C4 count using numpy vectorization"""
        A2 = A_mat @ A_mat
        # Only upper triangle
        iu = np.triu_indices(n, 1)
        vals = A2[iu]
        # C(k,2) = k*(k-1)/2 for k>=2, 0 otherwise
        c4_vals = np.where(vals >= 2, vals * (vals - 1) // 2, 0)
        return int(np.sum(c4_vals)) // 2
    
    def delta_c4_add_edge(A_mat, i, j, adj_sets):
        """Compute exact change in C4 count when adding edge (i,j).
        New C4s are those using edge (i,j): i-j-a-b-i where a~j, b~i, a~b, a!=b
        """
        cost = 0
        for a in adj_sets[j]:
            if a == i:
                continue
            for b in adj_sets[i]:
                if b == j or b == a:
                    continue
                if a in adj_sets[b]:
                    cost += 1
        return cost
    
    def delta_c4_remove_edge(A_mat, i, j, adj_sets):
        """Compute exact change in C4 count when removing edge (i,j).
        Lost C4s are those using edge (i,j): i-j-a-b-i where a~j, b~i, a~b, a!=b
        """
        cost = 0
        for a in adj_sets[j]:
            if a == i:
                continue
            for b in adj_sets[i]:
                if b == j or b == a:
                    continue
                if a in adj_sets[b]:
                    cost += 1
        return cost
    
    # Convert to sets for search
    adj_sets = [set() for _ in range(n)]
    for e in edges:
        adj_sets[e[0]].add(e[1])
        adj_sets[e[1]].add(e[0])
    
    # Edge set for O(1) lookup
    edge_set = set((min(e[0], e[1]), max(e[0], e[1])) for e in edges)
    
    # Count base 4-cycles
    base_4cycles = count_4cycles_fast(A)
    
    # Find the best non-edge to add (minimize new C4s)
    best_edge = None
    best_cost = float('inf')
    
    for i in range(n):
        for j in range(i+1, n):
            if (i, j) in edge_set:
                continue
            
            cost = delta_c4_add_edge(A, i, j, adj_sets)
            
            if cost < best_cost:
                best_cost = cost
                best_edge = (i, j)
            
            if time.time() > deadline - 60:
                break
        if time.time() > deadline - 60:
            break
    
    if best_edge is not None:
        u, v = best_edge
        edges.append([u, v])
        A[u, v] = 1
        A[v, u] = 1
        adj_sets[u].add(v)
        adj_sets[v].add(u)
        edge_set.add((u, v))
    
    current_count = count_4cycles_fast(A)
    
    # Local search with efficient delta computation
    improved = True
    max_iterations = 200
    iteration = 0
    no_improve_count = 0
    
    while improved and iteration < max_iterations and time.time() < deadline - 20:
        improved = False
        iteration += 1
        
        edge_list = list(edge_set)
        non_edges = []
        for i in range(n):
            for j in range(i+1, n):
                if (i, j) not in edge_set:
                    non_edges.append((i, j))
        
        best_swap = None
        best_net = 0  # we want most negative
        
        # Try all edge swaps with efficient delta computation
        for e1 in edge_list:
            if time.time() > deadline - 20:
                break
            u1, v1 = e1
            # Delta for removing e1
            delta_remove = -delta_c4_remove_edge(A, u1, v1, adj_sets)
            
            for e2 in non_edges:
                u2, v2 = e2
                # Skip if e2 shares vertices with e1 in problematic ways
                if u2 == u1 or u2 == v1 or v2 == u1 or v2 == v1:
                    # Still valid, but let's compute properly
                    pass
                
                # Delta for adding e2 (considering e1 is removed)
                # Need to compute delta_add with e1 removed
                # Temporarily remove e1 from adj_sets
                adj_sets[u1].remove(v1)
                adj_sets[v1].remove(u1)
                
                delta_add = delta_c4_add_edge(A, u2, v2, adj_sets)
                
                # Restore
                adj_sets[u1].add(v1)
                adj_sets[v1].add(u1)
                
                net = delta_remove + delta_add
                if net < best_net:
                    best_net = net
                    best_swap = (e1, e2)
        
        if best_swap is not None:
            e1, e2 = best_swap
            u1, v1 = e1
            u2, v2 = e2
            
            # Apply swap
            A[u1, v1] = 0
            A[v1, u1] = 0
            A[u2, v2] = 1
            A[v2, u2] = 1
            
            adj_sets[u1].remove(v1)
            adj_sets[v1].remove(u1)
            adj_sets[u2].add(v2)
            adj_sets[v2].add(u2)
            
            edge_set.remove(e1)
            edge_set.add(e2)
            
            edges = [list(e) for e in edge_set]
            improved = True
            no_improve_count = 0
        else:
            no_improve_count += 1
            if no_improve_count >= 3:
                break
    
    # Final validation
    final_edges = []
    seen = set()
    for e in edges:
        key = (min(e[0], e[1]), max(e[0], e[1]))
        if key not in seen:
            final_edges.append([e[0], e[1]])
            seen.add(key)
    
    # Ensure exactly 91 distinct edges
    if len(final_edges) > 91:
        final_edges = final_edges[:91]
    elif len(final_edges) < 91:
        for i in range(n):
            for j in range(i+1, n):
                if len(final_edges) >= 91:
                    break
                if (i, j) not in seen:
                    final_edges.append([i, j])
                    seen.add((i, j))
            if len(final_edges) >= 91:
                break
    
    return final_edges[:91]
