import numpy as np
import random
import time
import math

def construct():
    n = 16
    target_edges = 65
    start_time = time.time()
    deadline = start_time + 115.0  # Leave 5s margin for safety
    
    def calc_score(adj):
        degs = adj.sum(axis=1)
        max_sum = 0
        # Iterate over all triangles
        # A triangle is (i, j, k) with i < j < k
        for i in range(n):
            for j in range(i+1, n):
                if adj[i, j]:
                    for k in range(j+1, n):
                        if adj[i, k] and adj[j, k]:
                            s = degs[i] + degs[j] + degs[k]
                            if s > max_sum:
                                max_sum = s
        return max_sum

    def initial_graph():
        adj = np.zeros((n, n), dtype=bool)
        for u in range(8):
            for v in range(8, 16):
                adj[u, v] = True
                adj[v, u] = True
        # Add one edge inside left part
        adj[0, 1] = True
        adj[1, 0] = True
        return adj

    best_score = float('inf')
    best_adj = None
    
    # Run multiple restarts if time permits
    num_restarts = 0
    while time.time() < deadline:
        num_restarts += 1
        adj = initial_graph()
        
        # Perturb initial graph slightly to avoid identical restarts
        if num_restarts > 1:
            for _ in range(5):
                # Random swap
                u1 = random.randint(0, n-1)
                v1 = random.randint(0, n-1)
                if u1 == v1 or not adj[u1, v1]:
                    continue
                u2 = random.randint(0, n-1)
                v2 = random.randint(0, n-1)
                if u2 == v2 or adj[u2, v2]:
                    continue
                adj[u1, v1] = False
                adj[v1, u1] = False
                adj[u2, v2] = True
                adj[v2, u2] = True
        
        current_score = calc_score(adj)
        if current_score < best_score:
            best_score = current_score
            best_adj = adj.copy()
            
        T = 10.0
        cooling = 0.99995
        
        # Local search phase
        while time.time() < deadline:
            # Pick random edge to remove
            u1 = random.randint(0, n-1)
            v1 = random.randint(0, n-1)
            if u1 == v1 or not adj[u1, v1]:
                continue
                
            # Pick random non-edge to add
            u2 = random.randint(0, n-1)
            v2 = random.randint(0, n-1)
            if u2 == v2 or adj[u2, v2]:
                continue
            
            # Swap
            adj[u1, v1] = False
            adj[v1, u1] = False
            adj[u2, v2] = True
            adj[v2, u2] = True
            
            new_score = calc_score(adj)
            
            if new_score < current_score:
                current_score = new_score
                if new_score < best_score:
                    best_score = new_score
                    best_adj = adj.copy()
            else:
                delta = new_score - current_score
                if delta == 0:
                    accept = True
                else:
                    # Avoid overflow in exp
                    if delta / T > 50:
                        accept = False
                    else:
                        accept = random.random() < math.exp(-delta / T)
                
                if not accept:
                    # Revert
                    adj[u1, v1] = True
                    adj[v1, u1] = True
                    adj[u2, v2] = False
                    adj[v2, u2] = False
                else:
                    current_score = new_score
                    if new_score < best_score:
                        best_score = new_score
                        best_adj = adj.copy()
            
            T *= cooling
            if T < 0.1:
                break
        
        # If we haven't improved best_score in this restart, and we have time, continue
        # The outer while loop handles restarts

    # Reconstruct edge list from best_adj
    edges = []
    for i in range(n):
        for j in range(i+1, n):
            if best_adj[i, j]:
                edges.append([i, j])
                
    return {'n': n, 'edges': edges}
