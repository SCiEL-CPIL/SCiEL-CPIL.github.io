import numpy as np
import time

def heuristics(distance_matrix, coordinates, demands, capacity):
    start_time = time.time()
    deadline = start_time + 320.0
    
    D = np.asarray(distance_matrix, dtype=np.float64)
    n = D.shape[0]
    demands = np.asarray(demands, dtype=np.float64)
    
    def run_proxy_aco(gamma, num_ants=50):
        with np.errstate(divide='ignore', invalid='ignore'):
            eta = np.where(D > 0, D ** (-gamma), 1e-9)
        eta = np.where(np.isfinite(eta) & (eta > 0), eta, 1e-9)
        
        best_cost = float('inf')
        rng = np.random.RandomState(42) # Fixed seed for proxy consistency
        
        for _ in range(num_ants):
            unvisited = np.ones(n, dtype=bool)
            unvisited[0] = False
            current = 0
            load = 0.0
            total_dist = 0.0
            
            while unvisited.any():
                rem_cap = capacity - load
                feasible_mask = unvisited & (demands <= rem_cap)
                
                if not feasible_mask.any():
                    # Return to depot
                    total_dist += D[current, 0]
                    current = 0
                    load = 0.0
                    continue
                
                feasible_indices = np.where(feasible_mask)[0]
                weights = eta[current, feasible_indices]
                
                # Normalize weights
                sum_w = np.sum(weights)
                if sum_w <= 0:
                    next_node = rng.choice(feasible_indices)
                else:
                    probs = weights / sum_w
                    next_node = feasible_indices[rng.choice(len(feasible_indices), p=probs)]
                
                total_dist += D[current, next_node]
                load += demands[next_node]
                current = next_node
                unvisited[next_node] = False
                
                if not unvisited.any():
                    total_dist += D[current, 0]
                    break
            
            if total_dist < best_cost:
                best_cost = total_dist
        
        return best_cost

    # Grid search
    best_gamma = 1.0
    best_cost = float('inf')
    
    # Coarse search
    for gamma in np.arange(0.5, 3.1, 0.1):
        if time.time() > deadline - 10.0:
            break
        cost = run_proxy_aco(gamma, num_ants=50)
        if cost < best_cost:
            best_cost = cost
            best_gamma = gamma
            
    # Fine search around best_gamma
    if time.time() < deadline - 5.0:
        fine_gammas = np.arange(best_gamma - 0.2, best_gamma + 0.2, 0.02)
        for gamma in fine_gammas:
            if time.time() > deadline - 2.0:
                break
            if gamma < 0.1: continue
            cost = run_proxy_aco(gamma, num_ants=100) # More ants for fine search
            if cost < best_cost:
                best_cost = cost
                best_gamma = gamma

    # Construct final heuristic
    with np.errstate(divide='ignore', invalid='ignore'):
        final_heuristic = np.where(D > 0, D ** (-best_gamma), 1e-9)
    final_heuristic = np.where(np.isfinite(final_heuristic) & (final_heuristic > 0), final_heuristic, 1e-9)
    
    return final_heuristic
