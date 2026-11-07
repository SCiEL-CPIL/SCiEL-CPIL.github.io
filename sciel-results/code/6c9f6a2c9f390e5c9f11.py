import numpy as np
import time

def _run_aco_fast(distance_matrix, coordinates, demands, capacity, heuristic_matrix, seed, n_ants=20, n_iterations=30):
    """
    Fast ACO simulation using vectorized probability computation.
    Returns the best cost found.
    """
    np.random.seed(seed)
    n = distance_matrix.shape[0]
    pheromone = np.ones((n, n))
    
    best_cost_overall = float('inf')
    demands_arr = np.asarray(demands, dtype=np.float64)
    
    for _ in range(n_iterations):
        iteration_best_cost = float('inf')
        
        for _ in range(n_ants):
            current_node = 0
            current_load = 0.0
            unvisited = set(range(1, n))
            route_cost = 0.0
            
            while unvisited:
                # Find feasible neighbors
                feasible = [j for j in unvisited if demands_arr[j] <= capacity - current_load]
                
                if not feasible:
                    break
                
                # Calculate probabilities
                feasible_arr = np.array(feasible)
                probs = pheromone[current_node, feasible_arr] * heuristic_matrix[current_node, feasible_arr]
                
                total_prob = np.sum(probs)
                if total_prob == 0 or not np.isfinite(total_prob):
                    next_node = feasible[np.random.randint(len(feasible))]
                else:
                    probs = probs / total_prob
                    next_node = feasible[np.random.choice(len(feasible), p=probs)]
                
                # Move
                route_cost += distance_matrix[current_node, next_node]
                current_load += demands_arr[next_node]
                unvisited.remove(next_node)
                current_node = next_node
            
            # Return to depot
            route_cost += distance_matrix[current_node, 0]
            
            if route_cost < iteration_best_cost:
                iteration_best_cost = route_cost
        
        if iteration_best_cost < best_cost_overall:
            best_cost_overall = iteration_best_cost
            
        # Pheromone Update
        if iteration_best_cost > 0 and np.isfinite(iteration_best_cost):
            pheromone = 0.9 * pheromone + (1.0 / iteration_best_cost)
        else:
            pheromone = 0.9 * pheromone

    return best_cost_overall

def _run_aco_full(distance_matrix, coordinates, demands, capacity, heuristic_matrix, seed, n_ants=20, n_iterations=30):
    """
    Full-fidelity ACO simulation.
    Returns the best cost found.
    """
    np.random.seed(seed)
    n = distance_matrix.shape[0]
    pheromone = np.ones((n, n))
    
    best_cost_overall = float('inf')
    demands_arr = np.asarray(demands, dtype=np.float64)
    
    for _ in range(n_iterations):
        iteration_best_cost = float('inf')
        
        for _ in range(n_ants):
            current_node = 0
            current_load = 0.0
            unvisited = set(range(1, n))
            route_cost = 0.0
            
            while unvisited:
                feasible = [j for j in unvisited if demands_arr[j] <= capacity - current_load]
                
                if not feasible:
                    break
                
                feasible_arr = np.array(feasible)
                probs = pheromone[current_node, feasible_arr] * heuristic_matrix[current_node, feasible_arr]
                
                total_prob = np.sum(probs)
                if total_prob == 0 or not np.isfinite(total_prob):
                    next_node = feasible[np.random.randint(len(feasible))]
                else:
                    probs = probs / total_prob
                    next_node = feasible[np.random.choice(len(feasible), p=probs)]
                
                route_cost += distance_matrix[current_node, next_node]
                current_load += demands_arr[next_node]
                unvisited.remove(next_node)
                current_node = next_node
            
            route_cost += distance_matrix[current_node, 0]
            
            if route_cost < iteration_best_cost:
                iteration_best_cost = route_cost
        
        if iteration_best_cost < best_cost_overall:
            best_cost_overall = iteration_best_cost
            
        if iteration_best_cost > 0 and np.isfinite(iteration_best_cost):
            pheromone = 0.9 * pheromone + (1.0 / iteration_best_cost)
        else:
            pheromone = 0.9 * pheromone

    return best_cost_overall

def heuristics(distance_matrix, coordinates, demands, capacity):
    """
    Computes the heuristic matrix using multi-phase gamma optimization.
    """
    try:
        start_time = time.time()
        deadline = start_time + 320.0  # Safety margin before 330s limit
        
        dm = np.asarray(distance_matrix, dtype=np.float64)
        n = dm.shape[0]
        
        # Ensure diagonal is 0
        dm = dm.copy()
        np.fill_diagonal(dm, 0.0)
        
        # Phase 1: Coarse gamma search
        coarse_gammas = np.arange(0.3, 4.0, 0.2)
        best_gamma = 1.0
        best_score = float('inf')
        seeds = [42, 123, 456]
        
        for gamma in coarse_gammas:
            if time.time() > deadline:
                break
            
            with np.errstate(divide='ignore', invalid='ignore'):
                eta = 1.0 / (dm ** gamma)
            
            np.fill_diagonal(eta, 0.0)
            eta[~np.isfinite(eta)] = 1e-9
            eta[eta <= 0] = 1e-9
            
            # Use fast ACO for coarse search
            scores = []
            for seed in seeds:
                cost = _run_aco_fast(dm, coordinates, demands, capacity, eta, seed)
                scores.append(cost)
            
            mean_score = np.mean(scores)
            
            if mean_score < best_score:
                best_score = mean_score
                best_gamma = gamma
        
        # Phase 2: Fine gamma search around best
        if time.time() < deadline - 30:
            fine_gammas = np.arange(best_gamma - 0.2, best_gamma + 0.2, 0.05)
            fine_gammas = fine_gammas[fine_gammas > 0.1]
            
            for gamma in fine_gammas:
                if time.time() > deadline:
                    break
                
                with np.errstate(divide='ignore', invalid='ignore'):
                    eta = 1.0 / (dm ** gamma)
                
                np.fill_diagonal(eta, 0.0)
                eta[~np.isfinite(eta)] = 1e-9
                eta[eta <= 0] = 1e-9
                
                scores = []
                for seed in seeds:
                    cost = _run_aco_fast(dm, coordinates, demands, capacity, eta, seed)
                    scores.append(cost)
                
                mean_score = np.mean(scores)
                
                if mean_score < best_score:
                    best_score = mean_score
                    best_gamma = gamma
        
        # Phase 3: Ultra-fine polish with full-fidelity ACO
        if time.time() < deadline - 10:
            polish_gammas = np.arange(best_gamma - 0.05, best_gamma + 0.05, 0.01)
            polish_gammas = polish_gammas[polish_gammas > 0.1]
            
            for gamma in polish_gammas:
                if time.time() > deadline:
                    break
                
                with np.errstate(divide='ignore', invalid='ignore'):
                    eta = 1.0 / (dm ** gamma)
                
                np.fill_diagonal(eta, 0.0)
                eta[~np.isfinite(eta)] = 1e-9
                eta[eta <= 0] = 1e-9
                
                # Use full-fidelity ACO for polish
                scores = []
                for seed in seeds[:2]:  # Use fewer seeds for speed
                    cost = _run_aco_full(dm, coordinates, demands, capacity, eta, seed, n_ants=10, n_iterations=15)
                    scores.append(cost)
                
                mean_score = np.mean(scores)
                
                if mean_score < best_score:
                    best_score = mean_score
                    best_gamma = gamma
        
        # Final heuristic with best gamma
        with np.errstate(divide='ignore', invalid='ignore'):
            final_eta = 1.0 / (dm ** best_gamma)
        
        np.fill_diagonal(final_eta, 0.0)
        final_eta[~np.isfinite(final_eta)] = 1e-9
        final_eta[final_eta <= 0] = 1e-9
        
        return final_eta

    except Exception:
        # Fallback to standard heuristic if anything fails
        dm = np.asarray(distance_matrix, dtype=np.float64)
        dm = dm.copy()
        np.fill_diagonal(dm, 0.0)
        with np.errstate(divide='ignore', invalid='ignore'):
            eta = 1.0 / (dm + 1e-9)
        np.fill_diagonal(eta, 0.0)
        eta[~np.isfinite(eta)] = 1e-9
        eta[eta <= 0] = 1e-9
        return eta
