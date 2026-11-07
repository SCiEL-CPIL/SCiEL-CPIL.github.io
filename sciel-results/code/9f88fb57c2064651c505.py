# EVOLVE-BLOCK-START
import numpy as np
import time

def _aco_pretrain(distance_matrix, coordinates, demands, capacity, n_ants=10, n_iterations=10, seed=42):
    """
    Runs a lightweight ACO simulation to generate a pheromone matrix.
    This pheromone matrix will be used as the heuristic for the main evaluation.
    """
    n = distance_matrix.shape[0]
    np.random.seed(seed)
    
    # Initialize pheromone
    tau = np.ones((n, n))
    
    # Heuristic for the pretraining phase: standard 1/d
    eps = 1e-9
    safe_dist = distance_matrix + eps
    eta = 1.0 / safe_dist
    
    # Parameters
    alpha = 1.0
    beta = 2.0
    rho = 0.5  # Evaporation rate for pretraining (higher to explore more)
    
    best_cost_global = float('inf')
    
    for iter in range(n_iterations):
        # Check time limit
        if time.time() > 320: # Safety margin
            break
            
        # Initialize tours for ants
        tours = []
        
        for ant in range(n_ants):
            visited = np.zeros(n, dtype=bool)
            visited[0] = True
            current = 0
            remaining_capacity = capacity
            total_distance = 0.0
            tour = [0]
            
            unvisited_count = n - 1
            
            while unvisited_count > 0:
                # Find feasible next nodes
                feasible = []
                for j in range(1, n):
                    if not visited[j] and demands[j] <= remaining_capacity:
                        feasible.append(j)
                
                if not feasible:
                    # Return to depot
                    total_distance += distance_matrix[current, 0]
                    tour.append(0)
                    current = 0
                    remaining_capacity = capacity
                    continue
                
                # Calculate probabilities
                # prob ~ tau^alpha * eta^beta
                probs = []
                for j in feasible:
                    p = (tau[current, j] ** alpha) * (eta[current, j] ** beta)
                    probs.append(p)
                
                probs = np.array(probs)
                total_prob = np.sum(probs)
                if total_prob == 0:
                    # Fallback to uniform
                    probs = np.ones(len(feasible))
                    total_prob = len(feasible)
                
                probs = probs / total_prob
                
                # Roulette wheel selection
                r = np.random.random()
                cumulative = np.cumsum(probs)
                idx = np.searchsorted(cumulative, r)
                if idx >= len(feasible):
                    idx = len(feasible) - 1
                
                next_node = feasible[idx]
                
                total_distance += distance_matrix[current, next_node]
                remaining_capacity -= demands[next_node]
                visited[next_node] = True
                unvisited_count -= 1
                current = next_node
                tour.append(next_node)
            
            # Return to depot
            total_distance += distance_matrix[current, 0]
            tour.append(0)
            
            tours.append((tour, total_distance))
            
            if total_distance < best_cost_global:
                best_cost_global = total_distance
        
        # Update pheromone
        tau = (1 - rho) * tau
        
        for tour, cost in tours:
            # Only add pheromone if it's a reasonable solution
            # Using 1/cost as standard ACO
            delta_tau = 1.0 / cost if cost > 0 else 0.0
            
            # Add pheromone along the edges of the tour
            for k in range(len(tour) - 1):
                i = tour[k]
                j = tour[k + 1]
                tau[i, j] += delta_tau
                # Make it symmetric? The problem implies a directed graph for routing, 
                # but distance is symmetric. ACO usually updates directed edges.
                # However, the heuristic matrix returned is used in prob[i,j] = tau[i,j]*eta[i,j].
                # Since the ants traverse directed edges, we update directed pheromone.
        
        # Ensure pheromone is not zero
        tau = np.maximum(tau, 1e-9)
        
    return tau

def heuristics(distance_matrix, coordinates, demands, capacity):
    n = distance_matrix.shape[0]
    
    # Fallback heuristic: 1/d
    eps = 1e-9
    safe_dist = distance_matrix + eps
    fallback_eta = 1.0 / safe_dist
    
    try:
        # Run the pretraining ACO to get a learned heuristic (pheromone matrix)
        # We use a small number of ants/iterations to keep it fast
        learned_tau = _aco_pretrain(distance_matrix, coordinates, demands, capacity, n_ants=10, n_iterations=10, seed=42)
        
        # The learned pheromone matrix serves as the heuristic.
        # It reflects the "attractiveness" of edges based on the ACO's exploration.
        # We must ensure it's positive and finite.
        if not np.all(np.isfinite(learned_tau)):
            return fallback_eta
        
        # Normalize or scale? 
        # The ACO probability is tau * eta. If we return tau as eta, 
        # the final ACO will compute tau_final * eta_final = tau_initial * tau_learned.
        # This effectively squares the influence of the learned structure.
        # Alternatively, we can just return the learned tau as the heuristic.
        
        # Ensure positive
        learned_tau = np.where(learned_tau > 0, learned_tau, 1e-9)
        
        return learned_tau
        
    except Exception:
        # If anything goes wrong, fall back to standard 1/d
        return fallback_eta

# EVOLVE-BLOCK-END
