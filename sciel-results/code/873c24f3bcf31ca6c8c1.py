import time
import numpy as np
import random

def construct():
    start_time = time.time()
    deadline = start_time + 115
    
    n = 31
    target_edges = 91
    q = 5
    
    points = []
    for x in range(q):
        for y in range(q):
            points.append((1, x, y))
    for z in range(q):
        points.append((0, 1, z))
    points.append((0, 0, 1))
    
    def dot(p, r):
        return (p[0]*r[0] + p[1]*r[1] + p[2]*r[2]) % q
    
    adj = np.zeros((n, n), dtype=np.int8)
    for i in range(n):
        for j in range(i + 1, n):
            if dot(points[i], points[j]) == 0:
                adj[i, j] = 1
                adj[j, i] = 1
                
    def count_c4s(adj_matrix):
        A2 = adj_matrix @ adj_matrix
        total = 0
        for i in range(n):
            for j in range(i + 1, n):
                if adj_matrix[i, j] == 0:
                    k = A2[i, j]
                    if k >= 2:
                        total += k * (k - 1) // 2
        return total
    
    def count_new_c4s_add(u, v, adj_matrix):
        count = 0
        nu = np.where(adj_matrix[u] == 1)[0]
        nv = np.where(adj_matrix[v] == 1)[0]
        for a in nv:
            if a == u: continue
            for b in nu:
                if b == v or b == a: continue
                if adj_matrix[a, b] == 1:
                    count += 1
        return count

    # Add edges
    while np.sum(adj) // 2 < target_edges:
        if time.time() > deadline:
            break
        best_edge = None
        best_cost = float('inf')
        for i in range(n):
            for j in range(i + 1, n):
                if adj[i, j] == 0:
                    cost = count_new_c4s_add(i, j, adj)
                    if cost < best_cost:
                        best_cost = cost
                        best_edge = (i, j)
        if best_edge is None:
            break
        u, v = best_edge
        adj[u, v] = 1
        adj[v, u] = 1
        
    # Local Search: Best Improvement
    best_adj = adj.copy()
    best_c4s = count_c4s(adj)
    
    while time.time() < deadline:
        improved = False
        current_c4s = count_c4s(adj)
        
        edges_list = [(i, j) for i in range(n) for j in range(i+1, n) if adj[i,j] == 1]
        non_edges_list = [(i, j) for i in range(n) for j in range(i+1, n) if adj[i,j] == 0]
        
        best_swap = None
        best_new_c4s = current_c4s
        
        # Sample a subset of swaps if time is tight?
        # For n=31, full enumeration is ~34k swaps.
        # Each swap eval is ~30k ops. 34k * 30k = 1e9 ops.
        # This might be too slow in Python if done naively.
        # But count_c4s uses NumPy matrix mult, which is C-optimized.
        # 34k matrix mults of 31x31.
        # 31^3 = 30k flops. 34k * 30k = 1e9 flops.
        # NumPy can do ~1e9 flops/sec easily. So ~1s per full pass.
        # We have 115s. So we can do ~100 passes.
        
        for (u, v) in edges_list:
            for (x, y) in non_edges_list:
                if (u, v) == (x, y) or (u, v) == (y, x):
                    continue
                
                adj[u, v] = 0
                adj[v, u] = 0
                adj[x, y] = 1
                adj[y, x] = 1
                
                new_c4s = count_c4s(adj)
                
                if new_c4s < best_new_c4s:
                    best_new_c4s = new_c4s
                    best_swap = (u, v, x, y)
                    
                # Revert
                adj[u, v] = 1
                adj[v, u] = 1
                adj[x, y] = 0
                adj[y, x] = 0
                
                if time.time() > deadline:
                    break
            if time.time() > deadline:
                break
                
        if best_swap is not None:
            u, v, x, y = best_swap
            adj[u, v] = 0
            adj[v, u] = 0
            adj[x, y] = 1
            adj[y, x] = 1
            current_c4s = best_new_c4s
            improved = True
            
            if current_c4s < best_c4s:
                best_c4s = current_c4s
                best_adj = adj.copy()
        else:
            # No improvement found, try random restart or break
            # Since we are in a local minimum, we can try a random perturbation
            if time.time() > deadline - 10:
                break
            
            # Random swap
            if edges_list and non_edges_list:
                u, v = random.choice(edges_list)
                x, y = random.choice(non_edges_list)
                if (u, v) != (x, y) and (u, v) != (y, x):
                    adj[u, v] = 0
                    adj[v, u] = 0
                    adj[x, y] = 1
                    adj[y, x] = 1
                    # Don't update best_adj yet, just perturb to escape
                    continue
            else:
                break

    # Return the best found
    adj = best_adj
    result = []
    for i in range(n):
        for j in range(i + 1, n):
            if adj[i, j] == 1:
                result.append([i, j])
                
    return result
