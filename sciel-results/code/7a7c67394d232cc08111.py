import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 10s buffer

    n = 31
    
    # Step 1: Generate points of PG(2, 5)
    points = []
    seen = set()
    for x in range(5):
        for y in range(5):
            for z in range(5):
                if x == 0 and y == 0 and z == 0:
                    continue
                # Normalize
                if x != 0:
                    p = (1, (y * pow(x, 4, 5)) % 5, (z * pow(x, 4, 5)) % 5)
                elif y != 0:
                    p = (0, 1, (z * pow(y, 4, 5)) % 5)
                else:
                    p = (0, 0, 1)
                
                if p not in seen:
                    seen.add(p)
                    points.append(p)
    
    # Step 2: Construct Polarity Graph
    # Polarity: P ~ Q iff x1*x2 + y1*y2 - z1*z2 == 0 mod 5
    # -1 is 4 mod 5
    adj = [0] * n
    edges = []
    
    for i in range(n):
        p1 = points[i]
        for j in range(i + 1, n):
            p2 = points[j]
            val = (p1[0]*p2[0] + p1[1]*p2[1] + 4*p1[2]*p2[2]) % 5
            if val == 0:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
                edges.append((i, j))
                
    # Verify edge count
    if len(edges) != 90:
        # Fallback: if construction failed, use a random graph or simple structure
        # But it should be 90.
        pass

    # Function to count length-3 paths between u and v
    def count_paths_len3(u, v):
        count = 0
        # Neighbors of u and v
        # We iterate over all w in N(u) and x in N(v)
        # Check if w ~ x
        for w in range(n):
            if adj[u] & (1 << w):
                for x in range(n):
                    if adj[v] & (1 << x):
                        if w != x and w != v and x != u:
                            if adj[w] & (1 << x):
                                count += 1
        return count

    # Function to count total C4s
    def count_c4s():
        c4_count = 0
        for i in range(n):
            for j in range(i + 1, n):
                # Common neighbors of i and j
                common = adj[i] & adj[j]
                c = bin(common).count('1')
                c4_count += c * c
        return c4_count // 8

    # Initial C4 count should be 0
    current_c4s = count_c4s()
    
    # Step 3: Add 1 edge to minimize C4s
    best_add_edge = None
    min_c4_increase = float('inf')
    
    for i in range(n):
        for j in range(i + 1, n):
            if not (adj[i] & (1 << j)):
                inc = count_paths_len3(i, j)
                if inc < min_c4_increase:
                    min_c4_increase = inc
                    best_add_edge = (i, j)
                    
    if best_add_edge:
        u, v = best_add_edge
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        edges.append((u, v))
        current_c4s += min_c4_increase

    # Step 4: Local Search
    # Swap an existing edge with a non-existing edge to reduce C4s
    improved = True
    while improved and time.time() < deadline:
        improved = False
        best_swap = None
        best_delta = 0 # We want negative delta (reduction)
        
        # Get current edges and non-edges
        curr_edges = []
        curr_non_edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if adj[i] & (1 << j):
                    curr_edges.append((i, j))
                else:
                    curr_non_edges.append((i, j))
        
        # Precompute path counts for all edges and non-edges?
        # This might be expensive to recompute every iteration.
        # But n is small.
        
        # We can optimize by only checking promising swaps.
        # For now, check all.
        
        for e_rem in curr_edges:
            u, v = e_rem
            # C4s lost by removing (u,v)
            # This is the number of length-3 paths between u and v in the CURRENT graph.
            # Note: The path might use the edge (u,v)? No, length 3 path u-w-x-v does not use (u,v) directly unless w=u or x=v which is excluded.
            # However, the edge (u,v) is part of the graph.
            # If we remove (u,v), the C4s that disappear are those using (u,v).
            # A C4 using (u,v) is u-v-w-x-u.
            # This corresponds to a path u-x-w-v of length 3?
            # The path is u-x-w-v. Edges (u,x), (x,w), (w,v).
            # So yes, count_paths_len3(u, v) gives the number of C4s using (u,v).
            delta_remove = -count_paths_len3(u, v)
            
            for e_add in curr_non_edges:
                a, b = e_add
                # C4s gained by adding (a,b)
                # This is count_paths_len3(a, b) in the graph WITH (u,v) removed?
                # Approximation: count_paths_len3(a, b) in current graph.
                # If the path uses (u,v), it will be broken by removal, so we overestimate the gain.
                # This is a safe heuristic (might miss a swap, but won't accept a bad one if we verify).
                # Actually, if we overestimate gain, we might think delta is more negative than it is.
                # We should verify exact C4 count for the best candidate.
                
                delta_add = count_paths_len3(a, b)
                
                delta = delta_remove + delta_add
                
                if delta < best_delta:
                    best_delta = delta
                    best_swap = (e_rem, e_add)
                    
        if best_swap and best_delta < 0:
            e_rem, e_add = best_swap
            u, v = e_rem
            a, b = e_add
            
            # Apply swap
            adj[u] &= ~(1 << v)
            adj[v] &= ~(1 << u)
            adj[a] |= (1 << b)
            adj[b] |= (1 << a)
            
            # Verify exact C4 count
            new_c4s = count_c4s()
            if new_c4s < current_c4s:
                current_c4s = new_c4s
                improved = True
            else:
                # Revert if it didn't improve (shouldn't happen with exact check, but good for safety)
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                adj[a] &= ~(1 << b)
                adj[b] &= ~(1 << a)
                
    # Convert adj to list of edges
    final_edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if adj[i] & (1 << j):
                final_edges.append([i, j])
                
    return final_edges
