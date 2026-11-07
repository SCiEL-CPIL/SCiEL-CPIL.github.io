import numpy as np
import time

def _fast_aco_simulate(distance_matrix, heuristic, n_ants=5, n_iterations=10, rho=0.9):
    """
    Fast internal ACO simulation to evaluate heuristic quality.
    Correctly implements directed pheromone updates as per contract.
    """
    n = distance_matrix.shape[0]
    tau = np.ones((n, n))
    best_tour_len = float('inf')
    
    # Precompute log-probabilities? No, simple normalization is fine for N=50.
    
    for it in range(n_iterations):
        tour_lengths = []
        for ant in range(n_ants):
            visited = np.zeros(n, dtype=bool)
            visited[0] = True
            current = 0
            tour_len = 0.0
            tour_path = [0]
            
            for _ in range(n - 1):
                # Calculate probabilities for unvisited nodes
                # P[j] = tau[current, j] * heuristic[current, j] if not visited[j] else 0
                probs = tau[current, :] * heuristic[current, :]
                probs[visited] = 0.0
                
                total_prob = probs.sum()
                if total_prob < 1e-10:
                    # Fallback: pick random unvisited
                    unvisited_indices = np.where(~visited)[0]
                    next_node = unvisited_indices[np.random.randint(len(unvisited_indices))]
                else:
                    probs = probs / total_prob
                    # Use cumulative sum for selection
                    next_node = np.random.choice(n, p=probs)
                
                tour_len += distance_matrix[current, next_node]
                visited[next_node] = True
                tour_path.append(next_node)
                current = next_node
            
            # Close the tour
            tour_len += distance_matrix[current, 0]
            tour_lengths.append(tour_len)
            
            # Directed Pheromone Update: add 1/length to tau[i, j] for step i->j
            deposit = 1.0 / tour_len
            for i in range(n):
                u = tour_path[i]
                v = tour_path[(i + 1) % n]
                tau[u, v] += deposit
                # Do NOT update tau[v, u]
            
        # Global Evaporation
        tau *= rho
        
        if tour_lengths:
            current_best = min(tour_lengths)
            if current_best < best_tour_len:
                best_tour_len = current_best
                
    return best_tour_len

def heuristics(distance_matrix):
    start_time = time.time()
    deadline = start_time + 280.0 # Leave 20s margin for 3 instances total (approx 90s each)
    
    n = distance_matrix.shape[0]
    
    # Base heuristic: inverse distance
    safe_dist = distance_matrix.copy()
    np.fill_diagonal(safe_dist, 1.0) # Avoid division by zero, diagonal not used in tour anyway
    safe_dist[safe_dist < 1e-9] = 1e-9
    
    base_heur = 1.0 / safe_dist
    
    # Search for optimal beta
    # Based on observations, beta around 2.0 is good. Search 1.0 to 4.0.
    # Coarse search first
    betas = np.linspace(1.0, 4.0, 15)
    best_beta = 2.0
    best_score = float('inf')
    
    for beta in betas:
        if time.time() > deadline - 10.0: # Check deadline
            break
            
        h = base_heur ** beta
        # Ensure positive
        h[h <= 0] = 1e-9
        
        score = _fast_aco_simulate(distance_matrix, h, n_ants=5, n_iterations=10)
        
        if score < best_score:
            best_score = score
            best_beta = beta
            
    # Fine-grained search around best_beta
    if time.time() < deadline - 20.0:
        lower = max(1.0, best_beta - 0.5)
        upper = min(4.0, best_beta + 0.5)
        fine_betas = np.linspace(lower, upper, 10)
        
        for beta in fine_betas:
            if time.time() > deadline - 10.0:
                break
            h = base_heur ** beta
            h[h <= 0] = 1e-9
            score = _fast_aco_simulate(distance_matrix, h, n_ants=5, n_iterations=10)
            
            if score < best_score:
                best_score = score
                best_beta = beta
                
    # Construct final heuristic
    final_heuristic = base_heur ** best_beta
    final_heuristic[final_heuristic <= 0] = 1e-9
    
    # Final sanity check for non-finite values
    final_heuristic[~np.isfinite(final_heuristic)] = 1e-9
    
    return final_heuristic
