import numpy as np
import random
import time

def construct():
    n = 16
    # Initialize with seed: K_{8,8} + {0, 1}
    adj = np.zeros((n, n), dtype=np.int8)
    for u in range(8):
        for v in range(8, 16):
            adj[u, v] = 1
            adj[v, u] = 1
    adj[0, 1] = 1
    adj[1, 0] = 1
    
    # Precompute all potential triangles (i, j, k) with i < j < k
    triangles = []
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                triangles.append((i, j, k))
                
    def get_score(adj):
        degrees = adj.sum(axis=1)
        max_sum = 0
        # Vectorized check might be faster, but loop is fine for 560
        for i, j, k in triangles:
            if adj[i, j] and adj[j, k] and adj[i, k]:
                s = degrees[i] + degrees[j] + degrees[k]
                if s > max_sum:
                    max_sum = s
        return max_sum

    best_adj = adj.copy()
    best_score = get_score(best_adj)
    
    # All possible edges
    all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    
    start_time = time.time()
    timeout = 110 # Safety margin
    
    # Simulated Annealing Parameters
    T = 4.0
    cooling = 0.9995
    min_T = 0.1
    
    current_adj = best_adj.copy()
    current_score = best_score
    
    # To speed up random edge picking, we can maintain a list of edges and non-edges
    # But n is small, random.choice on 120 items is fast.
    
    while time.time() - start_time < timeout:
        if T < min_T:
            # Restart
            T = 4.0
            current_adj = best_adj.copy()
            current_score = best_score
        
        # Pick random edge to remove
        while True:
            e_remove = random.choice(all_edges)
            if current_adj[e_remove[0], e_remove[1]]:
                break
        
        # Pick random non-edge to add
        while True:
            e_add = random.choice(all_edges)
            if not current_adj[e_add[0], e_add[1]]:
                break
        
        u1, v1 = e_remove
        u2, v2 = e_add
        
        # Apply swap
        current_adj[u1, v1] = 0
        current_adj[v1, u1] = 0
        current_adj[u2, v2] = 1
        current_adj[v2, u2] = 1
        
        new_score = get_score(current_adj)
        
        if new_score < current_score:
            current_score = new_score
            if new_score < best_score:
                best_score = new_score
                best_adj = current_adj.copy()
        else:
            delta = new_score - current_score
            # Avoid overflow in exp
            if delta < 100 * T: 
                if random.random() < np.exp(-delta / T):
                    current_score = new_score
                else:
                    # Revert
                    current_adj[u1, v1] = 1
                    current_adj[v1, u1] = 1
                    current_adj[u2, v2] = 0
                    current_adj[v2, u2] = 0
            else:
                # Revert
                current_adj[u1, v1] = 1
                current_adj[v1, u1] = 1
                current_adj[u2, v2] = 0
                current_adj[v2, u2] = 0
        
        T *= cooling
    
    # Convert best_adj to list of edges
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if best_adj[i, j]:
                edges.append([i, j])
                
    return {'n': n, 'edges': edges}
