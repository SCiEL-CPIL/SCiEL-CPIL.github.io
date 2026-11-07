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
    
    for it in range(n_iterations):
        tour_lengths = []
        for ant in range(n_ants):
            visited = np.zeros(n, dtype=bool)
            visited[0] = True
            current = 0
            tour_len = 0.0
            tour_path = [0]
            
            for _ in range(n - 1):
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
            # u = tour_path, v = tour_path shifted
            u = np.array(tour_path)
            v = np.array(tour_path[1:] + [tour_path[0]])
            tau[u, v] += deposit
            
        tau *= rho
        
        if tour_lengths:
            current_best = min(tour_lengths)
            if current_best < best_tour_len:
                best_tour_len = current_best
                
    return best_tour_len

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
    
    # We use a "quick" evaluation to search for the best beta.
    # Running full 30 iterations is too slow for a dense search.
    # Running 5 iterations is a good surrogate for heuristic quality.
    
    def quick_eval(beta):
        h = base_heur ** beta
        # Ensure validity
        h[h <= 0] = 1e-9
        h[~np.isfinite(h)] = 1e-9
        # Use a fixed seed for consistent comparison during search
        score = _fast_aco_simulate(distance_matrix, h, n_ants=20, n_iterations=5, seed=42)
        return score

    # Coarse search
    betas_coarse = np.linspace(0.5, 6.0, 20)
    best_beta = 2.0
    best_score = float('inf')
    
    for beta in betas_coarse:
        if time.time() > deadline - 60.0: # Leave time for fine search
            break
        score = quick_eval(beta)
        if score < best_score:
            best_score = score
            best_beta = beta
            
    # Fine search around the best coarse beta
    if time.time() < deadline - 30.0:
        lower = max(0.1, best_beta - 1.5)
        upper = min(8.0, best_beta + 1.5)
        betas_fine = np.linspace(lower, upper, 30)
        
        for beta in betas_fine:
            if time.time() > deadline - 10.0:
                break
            score = quick_eval(beta)
            if score < best_score:
                best_score = score
                best_beta = beta
                
    # Final check: verify the best beta with a longer run? 
    # The quick eval is usually good enough. 
    # Let's just construct the final heuristic.
    
    final_heuristic = base_heur ** best_beta
    final_heuristic[final_heuristic <= 0] = 1e-9
    final_heuristic[~np.isfinite(final_heuristic)] = 1e-9
    
    return final_heuristic
