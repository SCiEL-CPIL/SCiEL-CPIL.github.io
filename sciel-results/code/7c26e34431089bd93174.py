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
                
    def count_c4(adj_mat):
        # Fast vectorized count
        A2 = np.dot(adj_mat, adj_mat)
        np.fill_diagonal(A2, 0)
        # Sum of C(k, 2) for all u != v is 2 * (Total C4s)
        # Wait, let's re-verify.
        # Sum_{u != v} C(k_uv, 2) = Sum_{u < v} C(k_uv, 2) + Sum_{v < u} C(k_vu, 2)
        # Since k_uv = k_vu, this is 2 * Sum_{u < v} C(k_uv, 2).
        # Total C4s = 0.5 * Sum_{u < v} C(k_uv, 2).
        # So Total C4s = 0.25 * Sum_{u != v} C(k_uv, 2).
        
        # Calculate Sum_{u != v} k*(k-1)
        # k*(k-1) = k^2 - k
        # Sum = sum(A2**2) - sum(A2)
        # Since diagonal is 0, sum(A2) is sum of off-diagonals.
        # sum(A2**2) is sum of squares of off-diagonals.
        
        s2 = np.sum(A2 * A2)
        s1 = np.sum(A2)
        total_pairs = s2 - s1
        # Total C4s = total_pairs / 4
        return total_pairs // 4

    current_edges = len(edges)
    
    # Adjust edge count to target_edges
    if current_edges > target_edges:
        # Remove edges
        to_remove = current_edges - target_edges
        # Greedy removal: remove edge that reduces C4s the most
        # Or random removal + local search
        # Let's try greedy first for a good start
        for _ in range(to_remove):
            best_edge = None
            best_c4 = float('inf')
            best_delta = 0
            for e in edges:
                u, v = e
                # Temporarily remove
                adj[u, v] = 0
                adj[v, u] = 0
                new_c4 = count_c4(adj)
                adj[u, v] = 1
                adj[v, u] = 1
                delta = new_c4 - count_c4(adj) # This is expensive, just use new_c4
                if new_c4 < best_c4:
                    best_c4 = new_c4
                    best_edge = e
            if best_edge:
                u, v = best_edge
                edges.remove(best_edge)
                adj[u, v] = 0
                adj[v, u] = 0
                
    elif current_edges < target_edges:
        # Add edges
        to_add = target_edges - current_edges
        for _ in range(to_add):
            best_edge = None
            best_c4 = float('inf')
            current_c4 = count_c4(adj)
            for u in range(n):
                for v in range(u + 1, n):
                    if adj[u, v] == 0:
                        adj[u, v] = 1
                        adj[v, u] = 1
                        new_c4 = count_c4(adj)
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
    # Perform Local Search / Simulated Annealing
    current_c4 = count_c4(adj)
    best_c4 = current_c4
    best_adj = adj.copy()
    
    temp = 1.0
    cooling = 0.999
    
    # Precompute all possible edges for faster sampling?
    # Or just sample random pairs.
    
    while time.time() < deadline:
        # Pick random edge to remove
        # Use list of edges for O(1) pick
        e_idx = random.randint(0, len(edges) - 1)
        u, v = edges[e_idx]
        
        # Pick random non-edge to add
        found = False
        attempts = 0
        while not found and attempts < 100:
            a = random.randint(0, n - 1)
            b = random.randint(0, n - 1)
            if a != b and adj[a, b] == 0:
                if not (a == u and b == v) and not (a == v and b == u):
                    found = True
            attempts += 1
            
        if not found:
            continue
            
        # Apply move
        adj[u, v] = 0
        adj[v, u] = 0
        adj[a, b] = 1
        adj[b, a] = 1
        
        new_c4 = count_c4(adj)
        
        if new_c4 < best_c4:
            best_c4 = new_c4
            best_adj = adj.copy()
            # Update edges list
            edges[e_idx] = (a, b)
            current_c4 = new_c4
        else:
            # Metropolis
            delta = new_c4 - current_c4
            if random.random() < np.exp(-delta / temp):
                edges[e_idx] = (a, b)
                current_c4 = new_c4
            else:
                # Revert
                adj[u, v] = 1
                adj[v, u] = 1
                adj[a, b] = 0
                adj[b, a] = 0
                
        temp *= cooling
        if temp < 0.01:
            temp = 0.01

    # Reconstruct final edges from best_adj
    final_edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if best_adj[i, j] == 1:
                final_edges.append([i, j])
                
    return final_edges
