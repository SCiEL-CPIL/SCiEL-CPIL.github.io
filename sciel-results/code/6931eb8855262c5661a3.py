import random
import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit
    
    n = 16
    target_edges = 65
    total_pairs = n * (n - 1) // 2  # 120
    
    # Precompute all possible edges
    all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    
    # Helper to calculate score efficiently
    def calc_score(adj, degs):
        max_sum = 0
        for u in range(n):
            for v in range(u + 1, n):
                if adj[u] & (1 << v):
                    common = adj[u] & adj[v]
                    while common:
                        lsb = common & -common
                        w = lsb.bit_length() - 1
                        s = degs[u] + degs[v] + degs[w]
                        if s > max_sum:
                            max_sum = s
                        common ^= lsb
        return max_sum
    
    def generate_random_graph():
        random.shuffle(all_edges)
        edges = all_edges[:target_edges]
        
        adj = [0] * n
        degs = [0] * n
        for u, v in edges:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            degs[u] += 1
            degs[v] += 1
        return adj, degs, edges
    
    def get_edges_from_adj(adj):
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if adj[i] & (1 << j):
                    edges.append([i, j])
        return edges
    
    best_score = float('inf')
    best_adj = None
    best_degs = None
    best_edges = None
    
    def update_best(adj, degs, score):
        nonlocal best_score, best_adj, best_degs, best_edges
        if score < best_score:
            best_score = score
            best_adj = adj[:]
            best_degs = degs[:]
            best_edges = get_edges_from_adj(adj)
    
    # Simulated annealing with restarts
    for restart in range(200):  # Max 200 restarts
        if time.time() > deadline:
            break
            
        adj, degs, edges = generate_random_graph()
        current_score = calc_score(adj, degs)
        update_best(adj, degs, current_score)
        
        # Simulated annealing parameters
        T = 10.0
        cooling_rate = 0.999
        no_improve_count = 0
        max_no_improve = 1000
        
        while time.time() < deadline and no_improve_count < max_no_improve:
            # Pick a random edge to remove
            while True:
                u1 = random.randint(0, n - 1)
                v1 = random.randint(0, n - 1)
                if u1 < v1 and (adj[u1] & (1 << v1)):
                    break
            
            # Pick a random non-edge to add
            while True:
                u2 = random.randint(0, n - 1)
                v2 = random.randint(0, n - 1)
                if u2 < v2 and not (adj[u2] & (1 << v2)):
                    break
            
            # Apply swap
            adj[u1] &= ~(1 << v1)
            adj[v1] &= ~(1 << u1)
            degs[u1] -= 1
            degs[v1] -= 1
            
            adj[u2] |= (1 << v2)
            adj[v2] |= (1 << u2)
            degs[u2] += 1
            degs[v2] += 1
            
            new_score = calc_score(adj, degs)
            delta = new_score - current_score
            
            # Accept if better, or with probability exp(-delta/T) if worse
            if delta <= 0 or random.random() < np.exp(-delta / T):
                current_score = new_score
                no_improve_count = 0
                update_best(adj, degs, current_score)
            else:
                # Revert
                adj[u1] |= (1 << v1)
                adj[v1] |= (1 << u1)
                degs[u1] += 1
                degs[v1] += 1
                
                adj[u2] &= ~(1 << v2)
                adj[v2] &= ~(1 << u2)
                degs[u2] -= 1
                degs[v2] -= 1
                
                no_improve_count += 1
            
            T *= cooling_rate
            if T < 0.1:
                T = 0.1
        
        # End local search phase, loop to next restart

    # Final polish: exhaustive single-swap search from best solution
    if best_adj is not None and time.time() < deadline - 10:
        adj = best_adj[:]
        degs = best_degs[:]
        current_score = best_score
        
        # Try all possible single swaps
        for u1 in range(n):
            for v1 in range(u1 + 1, n):
                if not (adj[u1] & (1 << v1)):
                    continue  # Not an edge
                for u2 in range(n):
                    for v2 in range(u2 + 1, n):
                        if (adj[u2] & (1 << v2)):
                            continue  # Already an edge
                        # Try swap
                        adj[u1] &= ~(1 << v1)
                        adj[v1] &= ~(1 << u1)
                        degs[u1] -= 1
                        degs[v1] -= 1
                        
                        adj[u2] |= (1 << v2)
                        adj[v2] |= (1 << u2)
                        degs[u2] += 1
                        degs[v2] += 1
                        
                        new_score = calc_score(adj, degs)
                        
                        if new_score < current_score:
                            current_score = new_score
                            update_best(adj, degs, current_score)
                        else:
                            # Revert
                            adj[u1] |= (1 << v1)
                            adj[v1] |= (1 << u1)
                            degs[u1] += 1
                            degs[v1] += 1
                            
                            adj[u2] &= ~(1 << v2)
                            adj[v2] &= ~(1 << u2)
                            degs[u2] -= 1
                            degs[v2] -= 1
            
            if time.time() > deadline - 10:
                break
        
        if time.time() > deadline - 10:
            break

    # Return the best found
    if best_edges is None:
        # Fallback to seed if search failed completely (shouldn't happen)
        edges = [[u, v] for u in range(8) for v in range(8, 16)] + [[0, 1]]
        return {'n': 16, 'edges': edges}
        
    return {'n': 16, 'edges': best_edges}
