import numpy as np
import time

def _nearest_neighbor_tour(d, start=0):
    n = d.shape[0]
    visited = np.zeros(n, dtype=bool)
    visited[start] = True
    tour = [start]
    current = start
    for _ in range(n - 1):
        # Find unvisited node with min distance
        dists = d[current].copy()
        dists[visited] = np.inf
        next_node = np.argmin(dists)
        tour.append(next_node)
        visited[next_node] = True
        current = next_node
    return tour

def _two_opt(d, tour):
    n = len(tour)
    improved = True
    while improved:
        improved = False
        for i in range(1, n - 1):
            for j in range(i + 1, n):
                # Check if edges (tour[i-1], tour[i]) and (tour[j], tour[(j+1)%n]) are the same or adjacent
                if i == j or i == j - 1:
                    continue
                a, b = tour[i-1], tour[i]
                c, e = tour[j], tour[(j+1) % n]
                # Current length: d(a,b) + d(c,e)
                # New length: d(a,c) + d(b,e) OR d(a,e) + d(b,c)
                # Standard 2-opt reversal: replace (a,b) and (c,e) with (a,c) and (b,e) 
                # Wait, standard 2-opt on path a-b-...-c-e is to reverse b-...-c.
                # New edges: (a, c) and (b, e).
                # Let's verify indices.
                # Tour: ... a, b, ..., c, e ...
                # Reverse segment b..c: ... a, c, ..., b, e ...
                # Edges removed: (a,b), (c,e)
                # Edges added: (a,c), (b,e)
                
                old_len = d[a, b] + d[c, e]
                new_len = d[a, c] + d[b, e]
                
                if new_len < old_len - 1e-10:
                    # Reverse tour[i:j+1]
                    tour[i:j+1] = tour[i:j+1][::-1]
                    improved = True
    return tour

def _simulate_aco(d, heuristic, n_ants=20, n_iter=30, seed=42):
    n = d.shape[0]
    np.random.seed(seed)
    
    h = heuristic.copy()
    h[~np.isfinite(h)] = 1e-9
    h[h <= 0] = 1e-9
    
    tau = np.ones((n, n))
    global_best_length = np.inf
    
    for it in range(n_iter):
        deposits = np.zeros((n, n))
        
        for ant in range(n_ants):
            visited = np.zeros(n, dtype=bool)
            visited[0] = True
            current = 0
            tour_length = 0.0
            tour = [0]
            
            for _ in range(n - 1):
                probs = tau[current, :] * h[current, :]
                probs[visited] = 0
                
                total = np.sum(probs)
                if total == 0:
                    unvisited_idx = np.where(~visited)[0]
                    next_node = np.random.choice(unvisited_idx)
                else:
                    probs = probs / total
                    next_node = np.random.choice(n, p=probs)
                
                tour_length += d[current, next_node]
                tour.append(next_node)
                visited[next_node] = True
                current = next_node
                
            tour_length += d[current, 0]
            tour.append(0)
            
            if tour_length < global_best_length:
                global_best_length = tour_length
            
            deposit_val = 1.0 / tour_length
            for k in range(len(tour) - 1):
                i = tour[k]
                j = tour[k+1]
                deposits[i, j] += deposit_val
        
        tau = tau * 0.9 + deposits
    
    return global_best_length

def heuristics(distance_matrix):
    start_time = time.time()
    deadline = 320.0
    
    d = np.asarray(distance_matrix, dtype=float)
    if not np.all(np.isfinite(d)):
        d = np.nan_to_num(d, nan=1.0, posinf=1.0, neginf=1.0)
    
    n = d.shape[0]
    
    # 1. Generate a high-quality approximate tour using NN + 2-Opt
    nn_tour = _nearest_neighbor_tour(d, 0)
    two_opt_tour = _two_opt(d, nn_tour)
    
    # Identify edges in the 2-Opt tour
    # The tour is a list of nodes [v0, v1, ..., v_{n-1}, v0]
    # Edges are (v_i, v_{i+1}) for i=0..n-1 (with v_n = v_0)
    tour_edges = set()
    for i in range(len(two_opt_tour) - 1):
        u, v = two_opt_tour[i], two_opt_tour[i+1]
        tour_edges.add((u, v))
        tour_edges.add((v, u))
        
    # Base heuristic: inverse distance
    inv_dist = 1.0 / (d + 1e-9)
    
    # Candidate boost factors
    boosts = [1.0, 10.0, 50.0, 100.0, 500.0, 1000.0, 5000.0]
    
    best_heuristic = inv_dist.copy()
    best_score = np.inf
    
    # We will evaluate each boost factor by running a quick ACO simulation
    # Using the exact parameters from the task (20 ants, 30 iters) to be accurate
    # But we can use a single seed for selection to save time, or multiple.
    # 7 boosts * 20 ants * 30 iters * 50 nodes * 50 prob calc = 10.5M ops.
    # This is very fast in Python/NumPy.
    
    seeds = [42] # Use one seed for selection to be fast, or [0,1,2] for robustness.
    # Let's use 3 seeds to be safe against variance.
    seeds = [0, 1, 2]
    
    for boost in boosts:
        if time.time() > deadline:
            break
            
        h_candidate = inv_dist.copy()
        for (u, v) in tour_edges:
            h_candidate[u, v] *= (1.0 + boost)
            
        # Ensure positive
        h_candidate[~np.isfinite(h_candidate)] = 1e-9
        h_candidate[h_candidate <= 0] = 1e-9
        
        total_score = 0.0
        for seed in seeds:
            score = _simulate_aco(d, h_candidate, n_ants=20, n_iter=30, seed=seed)
            total_score += score
            
        avg_score = total_score / len(seeds)
        
        if avg_score < best_score:
            best_score = avg_score
            best_heuristic = h_candidate
            
    return best_heuristic
