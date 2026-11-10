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
    
    def count_4cycles_np(A_mat):
        """Count 4-cycles using matrix multiplication: C4 = (A^2 * A^2 - diag terms) / 8
        Actually: number of 4-cycles = (sum_{i<j} C(common(i,j), 2)) / 2
        where common(i,j) = (A^2)[i,j] for i != j
        But we need to be careful: for adjacent i,j, common includes paths of length 2.
        The correct formula: for each pair (i,j), let c = (A^2)[i,j] - A[i,j] (exclude direct edge path)
        No wait, (A^2)[i,j] counts paths of length 2 from i to j, which are exactly common neighbors.
        So C4 = (1/2) * sum over all pairs (i,j) of C((A^2)[i,j], 2)
        But this counts each 4-cycle twice (once for each pair of opposite vertices).
        Actually no: a 4-cycle i-a-j-b-i has opposite pairs (i,j) and (a,b).
        For pair (i,j): common neighbors include a and b, so C(2,2)=1 contributes.
        For pair (a,b): common neighbors include i and j, so C(2,2)=1 contributes.
        So total = 2 per 4-cycle, and we divide by 2.
        """
        A2 = A_mat @ A_mat
        # For each pair (i,j), C(A2[i,j], 2)
        # C(k,2) = k*(k-1)/2
        # Sum over all i < j
        total = 0
        for i in range(n):
            for j in range(i+1, n):
                k = A2[i, j]
                if k >= 2:
                    total += k * (k - 1) // 2
        return total // 2
    
    def count_4cycles_fast(A_mat):
        """Faster C4 count using numpy vectorization"""
        A2 = A_mat @ A_mat
        # Only upper triangle
        iu = np.triu_indices(n, 1)
        vals = A2[iu]
        # C(k,2) = k*(k-1)/2 for k>=2, 0 otherwise
        c4_vals = np.where(vals >= 2, vals * (vals - 1) // 2, 0)
        return int(np.sum(c4_vals)) // 2
    
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
            
            # New C4s from adding edge (i,j):
            # 1. C4s where i,j are adjacent: for each pair of common neighbors a,b: i-a-j-b-i
            #    But wait, if i-j is an edge, then a 4-cycle containing edge ij is:
            #    i-j-a-b-i where a~j, b~i, a~b
            #    Or equivalently: for each common neighbor a of i and j, and each b~i, b~j... no.
            #    
            #    Let me use the general formula. Adding edge (i,j) changes (A^2) and thus C4 count.
            #    The change in C4 count when adding edge (i,j):
            #    - Pairs involving i or j as one endpoint: their common neighbor count changes
            #    - Specifically, for any vertex v, common(i,v) increases by 1 if v~j (new path i-j-v)
            #      and common(j,v) increases by 1 if v~i (new path j-i-v)
            #    - Also common(i,j) = number of common neighbors, which affects C(common,2)
            #    
            #    Let's just compute it directly:
            #    New C4s = C4s that use the new edge (i,j)
            #    A 4-cycle using edge (i,j) has form: i-j-a-b-i where a~j, b~i, a~b, a!=b, a!=i, b!=j
            #    Since i-j is now an edge, a cannot be i (a~j means a is neighbor of j, and j's neighbors don't include i before adding, but after adding... wait we're counting NEW cycles)
            #    Before adding, i is not a neighbor of j. After adding, i is.
            #    So a~j means a in adj[j] (old), and a != i (since i wasn't in adj[j] before)
            #    Similarly b~i means b in adj[i] (old), and b != j
            #    And a~b.
            #    
            #    So new C4s = |{(a,b) : a in adj[j], b in adj[i], a != b, a != i, b != j, a~b}|
            #    Since i not in adj[j] and j not in adj[i], the conditions a!=i and b!=j are automatic.
            #    
            #    Also: are there C4s where i and j are OPPOSITE vertices? No, because now they're adjacent.
            #    Before adding, if i,j had k common neighbors, there were C(k,2) C4s with i,j opposite.
            #    After adding, i,j are adjacent, so those C4s no longer have i,j as opposite.
            #    But those C4s still exist! They just have different opposite pairs.
            #    So the C4s with i,j opposite don't disappear; they're re-counted under different pairs.
            #    So we only need to count NEW C4s that use the edge (i,j).
            
            common = adj_sets[i] & adj_sets[j]
            # C4s where i,j are adjacent: i-j-a-b-i
            # a in adj[j], b in adj[i], a != b, a~b
            cost = 0
            for a in adj_sets[j]:
                if a == i:
                    continue
                for b in adj_sets[i]:
                    if b == j or b == a:
                        continue
                    if a in adj_sets[b]:
                        cost += 1
            
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
    
    # Local search with simulated annealing
    # For each edge swap (remove e1, add e2), compute the change in C4 count efficiently
    
    def delta_c4_remove_add(A_mat, u1, v1, u2, v2):
        """Compute change in C4 count when removing edge (u1,v1) and adding edge (u2,v2).
        Assumes (u1,v1) is currently an edge and (u2,v2) is currently a non-edge.
        """
        # We'll compute by temporarily modifying A and counting
        # But that's slow. Let's use a faster approach.
        # 
        # The C4 count is (1/2) * sum_{i<j} C((A^2)[i,j], 2)
        # Changing one edge affects (A^2) entries.
        # 
        # For simplicity and correctness, let's just recompute.
        # With n=31, A^2 is 31x31, which is fast.
        A_new = A_mat.copy()
        A_new[u1, v1] = 0
        A_new[v1, u1] = 0
        A_new[u2, v2] = 1
        A_new[v2, u2] = 1
        return count_4cycles_fast(A_new) - count_4cycles_fast(A_mat)
    
    # Systematic local search: try all edge swaps
    improved = True
    max_iterations = 100
    iteration = 0
    no_improve_count = 0
    
    while improved and iteration < max_iterations and time.time() < deadline - 30:
        improved = False
        iteration += 1
        
        current_count = count_4cycles_fast(A)
        
        edge_list = list(edge_set)
        non_edges = []
        for i in range(n):
            for j in range(i+1, n):
                if (i, j) not in edge_set:
                    non_edges.append((i, j))
        
        best_swap = None
        best_net = 0  # we want most negative
        
        # Try all edge swaps (systematic)
        for e1 in edge_list:
            if time.time() > deadline - 30:
                break
            u1, v1 = e1
            for e2 in non_edges:
                u2, v2 = e2
                # Skip if e2 shares a vertex with e1 in a way that creates issues
                # Actually no issue, just compute
                new_count = count_4cycles_fast(A.copy())
                # This is too slow. Let's do it properly.
                A_temp = A.copy()
                A_temp[u1, v1] = 0
                A_temp[v1, u1] = 0
                A_temp[u2, v2] = 1
                A_temp[v2, u2] = 1
                nc = count_4cycles_fast(A_temp)
                net = nc - current_count
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
