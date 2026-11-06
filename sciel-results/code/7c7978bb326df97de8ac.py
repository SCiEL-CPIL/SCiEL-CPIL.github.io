import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit
    
    n = 31
    target_edges = 91
    q = 5
    
    # 1. Construct Polarity Graph of PG(2, 5)
    points = []
    for x in range(q):
        for y in range(q):
            points.append((x, y, 1))
    for x in range(q):
        points.append((x, 1, 0))
    points.append((1, 0, 0))
    
    assert len(points) == 31
    
    adj = np.zeros((n, n), dtype=int)
    edges = []
    
    for i in range(n):
        for j in range(i + 1, n):
            p = points[i]
            q_pt = points[j]
            val = (p[0]*q_pt[0] + p[1]*q_pt[1] + p[2]*q_pt[2]) % q
            if val == 0:
                edges.append((i, j))
                adj[i, j] = 1
                adj[j, i] = 1
                
    def count_c4_fast(adj_mat):
        # #C4s = (1/2) * sum_{u < v} C(cn(u,v), 2) where cn(u,v) = (A^2)[u,v] for u != v
        A2 = np.dot(adj_mat, adj_mat)
        c2 = np.where(A2 >= 2, A2 * (A2 - 1) // 2, 0)
        c2[np.diag_indices(n)] = 0
        total = np.sum(c2) // 2
        return int(total)

    current_edges = len(edges)
    
    # Adjust edge count to target_edges
    if current_edges > target_edges:
        to_remove = current_edges - target_edges
        for _ in range(to_remove):
            best_edge = None
            best_c4 = float('inf')
            for e in edges:
                u, v = e
                adj[u, v] = 0
                adj[v, u] = 0
                new_c4 = count_c4_fast(adj)
                adj[u, v] = 1
                adj[v, u] = 1
                if new_c4 < best_c4:
                    best_c4 = new_c4
                    best_edge = e
            if best_edge:
                u, v = best_edge
                edges.remove(best_edge)
                adj[u, v] = 0
                adj[v, u] = 0
                
    elif current_edges < target_edges:
        to_add = target_edges - current_edges
        for _ in range(to_add):
            best_edge = None
            best_c4 = float('inf')
            for u in range(n):
                for v in range(u + 1, n):
                    if adj[u, v] == 0:
                        adj[u, v] = 1
                        adj[v, u] = 1
                        new_c4 = count_c4_fast(adj)
                        adj[u, v] = 0
                        adj[v, u] = 0
                        if new_c4 < best_c4:
                            best_c4 = new_c4
                            best_edge = (u, v)
            if best_edge:
                u, v = best_edge
                edges.append(best_edge)
                adj[u, v] = 1
                adj[v, u] = 1

    # Now we have exactly 91 edges.
    # Perform Local Search / Simulated Annealing with multiple restarts
    best_c4_overall = float('inf')
    best_adj_overall = None
    
    num_restarts = 5
    time_per_restart = (deadline - start_time) / num_restarts
    
    for restart in range(num_restarts):
        if time.time() > deadline - 5:
            break
            
        # Reset to base graph
        adj_r = np.zeros((n, n), dtype=int)
        edges_r = []
        for i in range(n):
            for j in range(i + 1, n):
                p = points[i]
                q_pt = points[j]
                val = (p[0]*q_pt[0] + p[1]*q_pt[1] + p[2]*q_pt[2]) % q
                if val == 0:
                    edges_r.append((i, j))
                    adj_r[i, j] = 1
                    adj_r[j, i] = 1
        
        # Find best edge to add
        best_edge = None
        best_c4_add = float('inf')
        for u in range(n):
            for v in range(u + 1, n):
                if adj_r[u, v] == 0:
                    adj_r[u, v] = 1
                    adj_r[v, u] = 1
                    new_c4 = count_c4_fast(adj_r)
                    adj_r[u, v] = 0
                    adj_r[v, u] = 0
                    if new_c4 < best_c4_add:
                        best_c4_add = new_c4
                        best_edge = (u, v)
        
        if best_edge:
            u, v = best_edge
            edges_r.append(best_edge)
            adj_r[u, v] = 1
            adj_r[v, u] = 1
        
        current_c4 = count_c4_fast(adj_r)
        best_c4 = current_c4
        best_adj_r = adj_r.copy()
        
        # Simulated Annealing
        temp = 1.0
        cooling = 0.9995
        restart_deadline = start_time + time_per_restart * (restart + 1)
        
        while time.time() < restart_deadline:
            e_idx = random.randint(0, len(edges_r) - 1)
            u, v = edges_r[e_idx]
            
            found = False
            attempts = 0
            while not found and attempts < 50:
                a = random.randint(0, n - 1)
                b = random.randint(0, n - 1)
                if a != b and adj_r[a, b] == 0:
                    if not (a == u and b == v) and not (a == v and b == u):
                        found = True
                attempts += 1
                
            if not found:
                continue
                
            adj_r[u, v] = 0
            adj_r[v, u] = 0
            adj_r[a, b] = 1
            adj_r[b, a] = 1
            
            new_c4 = count_c4_fast(adj_r)
            
            if new_c4 < best_c4:
                best_c4 = new_c4
                best_adj_r = adj_r.copy()
                edges_r[e_idx] = (a, b)
                current_c4 = new_c4
            else:
                delta = new_c4 - current_c4
                if delta == 0:
                    edges_r[e_idx] = (a, b)
                    current_c4 = new_c4
                elif random.random() < np.exp(-delta / temp):
                    edges_r[e_idx] = (a, b)
                    current_c4 = new_c4
                else:
                    adj_r[u, v] = 1
                    adj_r[v, u] = 1
                    adj_r[a, b] = 0
                    adj_r[b, a] = 0
                    
            temp *= cooling
            if temp < 0.01:
                temp = 0.01
        
        if best_c4 < best_c4_overall:
            best_c4_overall = best_c4
            best_adj_overall = best_adj_r.copy()
    
    if best_adj_overall is None:
        best_adj_overall = adj.copy()
    
    final_edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if best_adj_overall[i, j] == 1:
                final_edges.append([i, j])
                
    return final_edges
