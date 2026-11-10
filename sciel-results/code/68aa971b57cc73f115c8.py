import time
import numpy as np
import random

def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s limit
    
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
        """Count 4-cycles: for each pair (i,j), C(common_neighbors, 2) / 2
        Each C4 is counted twice (once for each diagonal pair)"""
        A2 = A_mat @ A_mat
        # Only upper triangle, exclude diagonal
        iu = np.triu_indices(n, 1)
        vals = A2[iu]
        # C(k,2) = k*(k-1)/2 for k>=2, 0 otherwise
        c4_vals = np.where(vals >= 2, vals * (vals - 1) // 2, 0)
        return int(np.sum(c4_vals)) // 2
    
    def delta_c4_add_edge(A_mat, i, j):
        """Compute exact change in C4 count when adding edge (i,j).
        New C4s use edge (i,j) as a diagonal or side.
        A C4 containing edge (i,j) as a side: i-j-a-b-i where a~j, b~i, a~b, a!=b, a!=i, b!=j
        Count: number of common neighbors of i and j that are NOT i or j themselves
        Actually: for each a in N(j)\{i}, for each b in N(i)\{j}, if a~b and a!=b, count 1
        """
        # Use numpy for speed
        Ai = A_mat[i]
        Aj = A_mat[j]
        # a must be neighbor of j (not i), b must be neighbor of i (not j)
        # a != b, a != i, b != j
        # Count pairs (a,b) where a in N(j)\{i}, b in N(i)\{j}, a!=b, and A[a,b]=1
        # Vectorized: for each a in N(j)\{i}, count b in N(i)\{j}\{a} with A[a,b]=1
        common = Ai & Aj  # neighbors of both i and j
        # But we need a~j, b~i, a~b, a!=b
        # Let's just do it with sets for clarity (n is small)
        adj_i = set(np.where(Ai)[0]) - {j}
        adj_j = set(np.where(Aj)[0]) - {i}
        cost = 0
        for a in adj_j:
            if a in adj_i:
                continue
            # a is neighbor of j but not i
            # b must be neighbor of i (not j), b != a, and a~b
            for b in adj_i:
                if b == a:
                    continue
                if A_mat[a, b] == 1:
                    cost += 1
        return cost
    
    def delta_c4_remove_edge(A_mat, i, j):
        """Same as add: the C4s lost are those using edge (i,j)"""
        return delta_c4_add_edge(A_mat, i, j)
    
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
            
            cost = delta_c4_add_edge(A, i, j)
            
            if cost < best_cost:
                best_cost = cost
                best_edge = (i, j)
    
    if best_edge is not None:
        u, v = best_edge
        edges.append([u, v])
        A[u, v] = 1
        A[v, u] = 1
        adj_sets[u].add(v)
        adj_sets[v].add(u)
        edge_set.add((u, v))
    
    current_count = count_4cycles_fast(A)
    
    # Local search: try edge swaps (remove one, add one)
    # Use numpy-based delta computation for speed
    improved = True
    max_iterations = 100
    iteration = 0
    no_improve_count = 0
    
    while improved and iteration < max_iterations and time.time() < deadline - 10:
        improved = False
        iteration += 1
        
        edge_list = list(edge_set)
        non_edges = []
        for i in range(n):
            for j in range(i+1, n):
                if (i, j) not in edge_set:
                    non_edges.append((i, j))
        
        best_swap = None
        best_net = 0  # we want most negative (decrease in C4 count)
        
        # For each edge to remove, compute its contribution
        # For each non-edge to add, compute its cost
        # Net = -delta_remove + delta_add (with both changes considered)
        
        # Precompute delta_remove for each edge
        delta_removes = {}
        for e1 in edge_list:
            u1, v1 = e1
            delta_removes[e1] = delta_c4_remove_edge(A, u1, v1)
        
        # Precompute delta_add for each non-edge
        delta_adds = {}
        for e2 in non_edges:
            u2, v2 = e2
            delta_adds[e2] = delta_c4_add_edge(A, u2, v2)
        
        # Find best swap
        for e1 in edge_list:
            if time.time() > deadline - 10:
                break
            dr = delta_removes[e1]
            for e2 in non_edges:
                da = delta_adds[e2]
                # Approximate net change: -dr + da
                # This is approximate because removing e1 changes the delta for adding e2
                # But it's a good heuristic
                net = -dr + da
                if net < best_net:
                    best_net = net
                    best_swap = (e1, e2)
        
        if best_swap is not None and best_net < 0:
            e1, e2 = best_swap
            u1, v1 = e1
            u2, v2 = e2
            
            # Verify with exact computation
            # Temporarily remove e1
            A[u1, v1] = 0
            A[v1, u1] = 0
            adj_sets[u1].remove(v1)
            adj_sets[v1].remove(u1)
            
            # Compute exact delta for adding e2
            exact_da = delta_c4_add_edge(A, u2, v2)
            
            # Restore
            A[u1, v1] = 1
            A[v1, u1] = 1
            adj_sets[u1].add(v1)
            adj_sets[v1].add(u1)
            
            exact_net = -delta_removes[e1] + exact_da
            
            if exact_net < 0:
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
        else:
            no_improve_count += 1
        
        if no_improve_count >= 5:
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
