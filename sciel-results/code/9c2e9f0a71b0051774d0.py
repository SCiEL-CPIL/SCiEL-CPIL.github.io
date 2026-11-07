import numpy as np
import time
import random

def heuristics(distance_matrix, coordinates, demands, capacity):
    """
    Returns an edge-attractiveness matrix for 3 fixed instances.
    Uses a grid search over the exponent gamma in (1/d)^gamma,
    evaluated by a fast, vectorized proxy ACO.
    """
    start_time = time.time()
    deadline = start_time + 320.0  # Safety margin before 330s kill
    
    D = np.asarray(distance_matrix, dtype=np.float64)
    n = D.shape[0]
    demands = np.asarray(demands, dtype=np.float64)
    
    # Candidate gamma values
    # Gamma > 1 focuses on local edges. Gamma < 1 allows longer jumps.
    candidate_gammas = np.arange(0.5, 3.5, 0.1)
    
    # Precompute demand feasibility masks for each possible current load?
    # Or just check on the fly. Since n=50, on the fly is fine.
    
    def run_proxy_aco(gamma, num_tours=100):
        """
        Fast proxy ACO using vectorized weight selection.
        Returns the best cost found.
        """
        # Construct heuristic matrix for this gamma
        with np.errstate(divide='ignore', invalid='ignore'):
            eta = np.where(D > 0, D ** (-gamma), 1e-9)
        
        # Sanitize: any non-finite or non-positive becomes 1e-9
        eta = np.where(np.isfinite(eta) & (eta > 0), eta, 1e-9)
        
        best_cost = float('inf')
        rng = np.random.RandomState(int(gamma * 100)) # Seed per gamma for reproducibility in proxy
        
        for t in range(num_tours):
            # Check time
            if time.time() > deadline:
                break
            
            current = 0
            current_load = 0.0
            total_distance = 0.0
            unvisited = np.ones(n, dtype=bool)
            unvisited[0] = False
            
            # Pre-compute indices of unvisited customers for faster iteration?
            # We need to dynamically remove them.
            # Using a list of indices is faster than boolean mask + np.where every step
            # if we can maintain it. But removing from a list is O(n).
            # With n=50, it's negligible.
            
            unvisited_indices = list(range(1, n))
            
            while unvisited_indices:
                # Find feasible neighbors
                feasible_indices = []
                feasible_weights = []
                
                # Vectorized feasibility check
                # demands[unvisited_indices] <= capacity - current_load
                # We can filter the list
                rem_cap = capacity - current_load
                for j in unvisited_indices:
                    if demands[j] <= rem_cap:
                        feasible_indices.append(j)
                
                if not feasible_indices:
                    # Return to depot
                    total_distance += D[current, 0]
                    break
                
                feasible_arr = np.array(feasible_indices)
                weights = eta[current, feasible_arr]
                
                total_w = np.sum(weights)
                if total_w <= 0:
                    next_node = rng.choice(feasible_indices)
                else:
                    probs = weights / total_w
                    # rng.choice with p
                    next_node = feasible_arr[rng.choice(len(feasible_arr), p=probs)]
                
                # Move
                total_distance += D[current, next_node]
                current_load += demands[next_node]
                
                # Remove next_node from unvisited
                # List remove is O(n). n=50.
                unvisited_indices.remove(next_node)
                current = next_node
                
                if not unvisited_indices:
                    # All visited, return to depot
                    total_distance += D[current, 0]
                    break
            
            if total_distance < best_cost:
                best_cost = total_distance
        
        return best_cost

    best_gamma = 1.0
    best_proxy_cost = float('inf')
    
    # Search over gammas
    # We have ~30 gammas.
    # Let's run a quick pass to see how long one gamma takes.
    # If it's too slow, we reduce num_tours or gammas.
    
    # Estimate time per gamma
    # If we have 320s and 30 gammas, we have ~10s per gamma.
    # 100 tours * 50 steps * (list ops + numpy small ops) should be very fast (<0.1s).
    # So 100 tours is safe. Maybe even 200.
    
    num_tours_per_gamma = 100
    
    for gamma in candidate_gammas:
        if time.time() > deadline - 5.0: # 5s buffer for final matrix construction
            break
            
        cost = run_proxy_aco(gamma, num_tours=num_tours_per_gamma)
        
        if cost < best_proxy_cost:
            best_proxy_cost = cost
            best_gamma = gamma
            
            # Optional: Refine around best_gamma?
            # If we have time, we can do a finer search around the current best.
            # But the grid is already 0.1 step.
            # Let's just stick to the grid to avoid complexity and potential bugs.
            
    # Construct final heuristic matrix with best_gamma
    with np.errstate(divide='ignore', invalid='ignore'):
        final_heuristic = np.where(D > 0, D ** (-best_gamma), 1e-9)
    
    # Sanitize
    final_heuristic = np.where(np.isfinite(final_heuristic) & (final_heuristic > 0), final_heuristic, 1e-9)
    
    # Ensure shape is correct
    if final_heuristic.shape != D.shape:
        # Should not happen, but safety check
        final_heuristic = np.ones_like(D) * 1e-9
        np.fill_diagonal(final_heuristic, 1e-9)
        
    return final_heuristic
