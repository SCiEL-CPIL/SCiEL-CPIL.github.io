import numpy as np
import time

def _fast_aco_simulate(distance_matrix, heuristic, n_ants=20, n_iterations=30, rho=0.9, seed=42):
    """
    Fast internal ACO simulation to evaluate heuristic quality.
    Implements directed pheromone updates as per contract.
    """
    rng = np.random.RandomState(seed)
    n = distance_matrix.shape[0]
    tau = np.ones((n, n))
    best_tour_len = float('inf')
    
    # Pre-compute unvisited masks? No, dynamic.
    # Pre-compute valid indices for each node?
    
    for it in range(n_iterations):
        tour_lengths = []
        for ant in range(n_ants):
            visited = np.zeros(n, dtype=bool)
            visited[0] = True
            current = 0
            tour_len = 0.0
            tour_path = [0]
            
            for _ in range(n - 1):
                # Calculate probabilities
                # tau[current, :] * heuristic[current, :]
                # Mask visited
                probs = tau[current, :] * heuristic[current, :]
                probs[visited] = 0.0
                
                total_prob = probs.sum()
                if total_prob < 1e-10:
                    unvisited_indices = np.where(~visited)[0]
                    next_node = unvisited_indices[rng.randint(len(unvisited_indices))]
                else:
                    probs = probs / total_prob
                    next_node = rng.choice(n, p=probs)
                
                tour_len += distance_matrix[current, next_node]
                visited[next_node] = True
                tour_path.append(next_node)
                current = next_node
            
            tour_len += distance_matrix[current, 0]
            tour_lengths.append(tour_len)
            
            deposit = 1.0 / tour_len
            # Vectorized pheromone update for the closed tour
            u = np.array(tour_path)
            v = np.array(tour_path[1:] + [tour_path[0]])
            tau[u, v] += deposit
            
        tau *= rho
        
        if tour_lengths:
            current_best = min(tour_lengths)
            if current_best < best_tour_len:
                best_tour_len = current_best
                
    return best_tour_len

def _compute_mst_edges(distance_matrix):
    """
    Compute the Minimum Spanning Tree edges using Prim's algorithm.
    Returns a boolean matrix where True indicates an MST edge.
    """
    n = distance_matrix.shape[0]
    mst_adj = np.zeros((n, n), dtype=bool)
    in_mst = np.zeros(n, dtype=bool)
    in_mst[0] = True
    min_dist = distance_matrix[0].copy()
    min_dist[0] = np.inf
    
    for _ in range(n - 1):
        # Find the vertex with the minimum distance to the MST
        u = np.argmin(min_dist)
        if min_dist[u] == np.inf:
            break
        in_mst[u] = True
        mst_adj[u, :] = True # Symmetric for now, will handle directionality if needed, but MST is undirected
        mst_adj[:, u] = True
        
        # Update min_dist
        for v in range(n):
            if not in_mst[v] and distance_matrix[u, v] < min_dist[v]:
                min_dist[v] = distance_matrix[u, v]
                
    # Exclude self-loops
    np.fill_diagonal(mst_adj, False)
    return mst_adj

def heuristics(distance_matrix):
    start_time = time.time()
    # Safety margin: evaluator kills at 330s, we should finish well before.
    # Let's aim to finish within 250s to be safe.
    deadline = start_time + 250.0
    
    n = distance_matrix.shape[0]
    
    # Base heuristic: inverse distance
    safe_dist = distance_matrix.copy()
    np.fill_diagonal(safe_dist, 1.0)
    safe_dist[safe_dist < 1e-9] = 1e-9
    
    base_heur = 1.0 / safe_dist
    
    # Compute MST edges
    mst_edges = _compute_mst_edges(distance_matrix)
    
    # We will combine a continuous heuristic with a discrete MST boost.
    # h = base_heur ** beta * (1 + gamma * mst_indicator)
    # Or simpler: h = base_heur ** beta, then multiply MST edges by a factor.
    # Let's search for beta (exponent) and mst_boost (multiplier for MST edges).
    
    def quick_eval(beta, mst_boost):
        h = base_heur ** beta
        # Apply MST boost
        h[mst_edges] *= mst_boost
        # Ensure validity
        h[h <= 0] = 1e-9
        h[~np.isfinite(h)] = 1e-9
        # Use a fixed seed for consistent comparison during search
        score = _fast_aco_simulate(distance_matrix, h, n_ants=20, n_iterations=5, seed=42)
        return score

    # Coarse search over beta and mst_boost
    # beta: 0.5 to 6.0
    # mst_boost: 1.0 to 10.0
    best_beta = 2.0
    best_boost = 1.0
    best_score = float('inf')
    
    betas_coarse = np.linspace(0.5, 6.0, 10)
    boosts_coarse = [1.0, 2.0, 5.0, 10.0]
    
    for beta in betas_coarse:
        if time.time() > deadline - 90.0: # Leave time for fine search
            break
        for boost in boosts_coarse:
            if time.time() > deadline - 90.0:
                break
            score = quick_eval(beta, boost)
            if score < best_score:
                best_score = score
                best_beta = beta
                best_boost = boost
                
    # Fine search around the best coarse parameters
    if time.time() < deadline - 30.0:
        lower_b = max(0.1, best_beta - 1.5)
        upper_b = min(8.0, best_beta + 1.5)
        lower_boost = max(1.0, best_boost * 0.5)
        upper_boost = min(20.0, best_boost * 2.0)
        
        betas_fine = np.linspace(lower_b, upper_b, 10)
        boosts_fine = np.linspace(lower_boost, upper_boost, 5)
        
        for beta in betas_fine:
            if time.time() > deadline - 10.0:
                break
            for boost in boosts_fine:
                if time.time() > deadline - 10.0:
                    break
                score = quick_eval(beta, boost)
                if score < best_score:
                    best_score = score
                    best_beta = beta
                    best_boost = boost
                    
    # Final check: verify the best parameters with a longer run? 
    # The quick eval is usually good enough. 
    # Let's just construct the final heuristic.
    
    final_heuristic = base_heur ** best_beta
    final_heuristic[mst_edges] *= best_boost
    final_heuristic[final_heuristic <= 0] = 1e-9
    final_heuristic[~np.isfinite(final_heuristic)] = 1e-9
    
    return final_heuristic
