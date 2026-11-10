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
        
    # Convert to sets for search
    adj_sets = [set() for _ in range(n)]
    for e in edges:
        adj_sets[e[0]].add(e[1])
        adj_sets[e[1]].add(e[0])
    
    # Edge set for O(1) lookup
    edge_set = set((min(e[0], e[1]), max(e[0], e[1])) for e in edges)
    
    def count_4cycles_fast(A_mat):
        """Faster C4 count using numpy vectorization"""
        A2 = A_mat @ A_mat
        # Only upper triangle
        iu = np.triu_indices(n, 1)
        vals = A2[iu]
        # C(k,2) = k*(k-1)/2 for k>=2, 0 otherwise
        c4_vals = np.where(vals >= 2, vals * (vals - 1) // 2, 0)
        return int(np.sum(c4_vals)) // 2

    # We have 90 edges. We need 91.
    # Add an edge. Let's pick one that minimizes new C4s.
    # New C4s from adding (u,v) = number of edges (w,x) such that w in N(u), x in N(v), w != x.
    # Actually, a C4 is u-w-x-v-u.
    # So we need w in N(u), x in N(v), and (w,x) in E.
    
    best_add_edge = None
    min_new_c4s = float('inf')
    
    for i in range(n):
        for j in range(i+1, n):
            if (i, j) in edge_set:
                continue
            
            # Count new C4s
            new_c4s = 0
            # Iterate over neighbors of i
            for w in adj_sets[i]:
                # Iterate over neighbors of j
                for x in adj_sets[j]:
                    if w == x:
                        continue
                    # Check if (w,x) is an edge
                    if w < x:
                        if (w, x) in edge_set:
                            new_c4s += 1
                    else:
                        if (x, w) in edge_set:
                            new_c4s += 1
            
            if new_c4s < min_new_c4s:
                min_new_c4s = new_c4s
                best_add_edge = (i, j)
                
    # Add the best edge
    if best_add_edge:
        u, v = best_add_edge
        edges.append([u, v])
        A[u, v] = 1
        A[v, u] = 1
        adj_sets[u].add(v)
        adj_sets[v].add(u)
        edge_set.add((u, v))
        
    current_count = count_4cycles_fast(A)
    
    # Local Search: Swap edges to reduce C4 count
    # Delta for adding edge (u,v):
    # New C4s formed: u-w-x-v-u where w~u, x~v, w!=x, w~x.
    # Note: w~x must be an edge in the CURRENT graph.
    
    def delta_add(u, v):
        # u, v are non-adjacent
        # Count pairs (w, x) such that w in N(u), x in N(v), w != x, and (w,x) in E
        count = 0
        # Optimization: iterate over smaller set?
        for w in adj_sets[u]:
            for x in adj_sets[v]:
                if w == x:
                    continue
                if w < x:
                    if (w, x) in edge_set:
                        count += 1
                else:
                    if (x, w) in edge_set:
                        count += 1
        return count

    def delta_remove(u, v):
        # u, v are adjacent
        # Count C4s destroyed: u-w-x-v-u where w in N(u)\{v}, x in N(v)\{u}, w!=x, w~x.
        # Note: w~x must be an edge in the CURRENT graph.
        count = 0
        for w in adj_sets[u]:
            if w == v:
                continue
            for x in adj_sets[v]:
                if x == u:
                    continue
                if w == x:
                    continue
                if w < x:
                    if (w, x) in edge_set:
                        count += 1
                else:
                    if (x, w) in edge_set:
                        count += 1
        return count

    # Perform swaps
    # We want to find (e1, e2) such that delta_remove(e1) - delta_add(e2) > 0 (decrease in C4s)
    # Actually, change = -delta_remove(e1) + delta_add(e2). We want change < 0.
    # So we want delta_remove(e1) > delta_add(e2).
    
    # Precompute costs?
    # Costs change as graph changes.
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        best_swap = None
        best_gain = 0 # We want to maximize gain (decrease in C4 count)
        
        # List current edges and non-edges
        current_edges = list(edge_set)
        non_edges = []
        for i in range(n):
            for j in range(i+1, n):
                if (i, j) not in edge_set:
                    non_edges.append((i, j))
        
        # To speed up, we can sample or limit checks if too many
        # But n=31 is small. 91 edges, ~389 non-edges.
        # 91 * 389 ~ 35k swaps.
        # Each swap check is O(deg^2). Deg ~ 6. So 36 ops.
        # 35k * 36 ~ 1.2M ops. Very fast in Python? Maybe 1-2 seconds per full pass.
        
        for e1 in current_edges:
            u1, v1 = e1
            cost_remove = delta_remove(u1, v1)
            
            if cost_remove <= best_gain:
                continue # Can't beat current best gain with this removal
            
            for e2 in non_edges:
                u2, v2 = e2
                # Skip if e2 is same as e1 (not possible as e1 in edge_set, e2 not)
                cost_add = delta_add(u2, v2)
                
                gain = cost_remove - cost_add
                if gain > best_gain:
                    best_gain = gain
                    best_swap = (e1, e2)
        
        if best_swap:
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
            
            current_count -= best_gain
            improved = True
    
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
