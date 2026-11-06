import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    
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
        A2 = np.dot(adj_mat, adj_mat)
        c2 = np.where(A2 >= 2, A2 * (A2 - 1) // 2, 0)
        c2[np.diag_indices(n)] = 0
        total = np.sum(c2) // 2
        return int(total)

    current_edges = len(edges)
    # Polarity graph of PG(2,5) has 93 edges.
    # We need 91.
    # Step 1: Remove 3 edges to get 90 edges, minimizing C4s.
    # Step 2: Add 1 edge to get 91 edges, minimizing C4s.
    
    # Phase 1: Remove edges to reach 90
    while len(edges) > 90:
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
        else:
            break

    # Now we have 90 edges.
    # Phase 2: Add 1 edge to reach 91
    if len(edges) < 91:
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

    # Final Local Search on the 91-edge graph
    best_c4_overall = count_c4_fast(adj)
    best_adj_overall = adj.copy()
    
    # Simulated Annealing / Local Search
    temp = 1.0
    cooling = 0.999
    current_c4 = best_c4_overall
    
    while time.time() < deadline - 5:
        # Pick a random edge to remove
        e_idx = random.randint(0, len(edges) - 1)
        u, v = edges[e_idx]
        
        # Pick a random non-edge to add
        found = False
        attempts = 0
        while not found and attempts < 100:
            a = random.randint(0, n - 1)
            b = random.randint(0, n - 1)
            if a != b and adj[a, b] == 0 and not (a == u and b == v) and not (a == v and b == u):
                found = True
            attempts += 1
            
        if not found:
            continue
            
        adj[u, v] = 0
        adj[v, u] = 0
        adj[a, b] = 1
        adj[b, a] = 1
        
        new_c4 = count_c4_fast(adj)
        
        if new_c4 < best_c4_overall:
            best_c4_overall = new_c4
            best_adj_overall = adj.copy()
            edges[e_idx] = (a, b)
            current_c4 = new_c4
        else:
            delta = new_c4 - current_c4
            if delta == 0:
                edges[e_idx] = (a, b)
                current_c4 = new_c4
            elif random.random() < np.exp(-delta / temp):
                edges[e_idx] = (a, b)
                current_c4 = new_c4
            else:
                adj[u, v] = 1
                adj[v, u] = 1
                adj[a, b] = 0
                adj[b, a] = 0
                
        temp *= cooling
        if temp < 0.01:
            temp = 0.01

    final_edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if best_adj_overall[i, j] == 1:
                final_edges.append([i, j])
                
    return final_edges
