"""Heuristic for ACO-CVRP: Cluster-Aware Heuristic with Local Search Optimization."""
import numpy as np
import time
import random

_T0 = time.time()
_BUDGET = 325.0  # Leave 5s safety margin
_DEADLINE = _T0 + _BUDGET

def _remaining():
    return max(0.0, _DEADLINE - time.time())

def _run_aco_simulation(distance_matrix, demands, capacity, heuristic, n_ants=10, n_iter=5, seed=42):
    """
    Run a fast ACO simulation to evaluate a heuristic.
    Returns the best total route distance found by any ant.
    """
    np.random.seed(seed)
    n = distance_matrix.shape[0]
    
    # Pheromone starts at 1
    pheromone = np.ones((n, n))
    
    best_cost_ever = float('inf')
    
    # Pre-compute feasible neighbors for each node? 
    # Since capacity is fixed, feasible neighbors depend on current load, which varies.
    # We must check on the fly.
    
    for iter_i in range(n_iter):
        iteration_costs = []
        
        for ant_i in range(n_ants):
            node = 0  # Start at depot
            remaining_load = capacity
            cost = 0.0
            visited = np.zeros(n, dtype=bool)
            visited[0] = True
            
            while True:
                # Find all unvisited customers that fit in remaining load
                feasible = []
                # Optimization: Vectorized check for feasibility
                # demands[1:] <= remaining_load and not visited[1:]
                feasible_mask = (~visited[1:]) & (demands[1:] <= remaining_load + 1e-9)
                feasible_indices = np.where(feasible_mask)[0] + 1
                
                if len(feasible_indices) == 0:
                    # Return to depot and reload
                    cost += distance_matrix[node, 0]
                    break
                
                # Calculate probabilities: pheromone[i,j] * heuristic[i,j]
                # Extract relevant pheromone and heuristic values
                p_vals = pheromone[node, feasible_indices]
                h_vals = heuristic[node, feasible_indices]
                probs = p_vals * h_vals
                
                total_prob = np.sum(probs)
                
                if total_prob <= 1e-15:
                    # Fallback: uniform random
                    idx = np.random.randint(len(feasible_indices))
                else:
                    probs = probs / total_prob
                    r = np.random.random()
                    cumulative = np.cumsum(probs)
                    idx = np.searchsorted(cumulative, r)
                    if idx >= len(feasible_indices):
                        idx = len(feasible_indices) - 1
                
                next_node = feasible_indices[idx]
                cost += distance_matrix[node, next_node]
                remaining_load -= demands[next_node]
                visited[next_node] = True
                node = next_node
        
        iteration_costs.append(cost)
        
        best_this_iter = min(iteration_costs)
        if best_this_iter < best_cost_ever:
            best_cost_ever = best_this_iter
        
        # Pheromone update: evaporate then add
        pheromone *= 0.9
        pheromone += 1.0 / best_this_iter
    
    return best_cost_ever

def _construct_heuristic(distance_matrix, demands, capacity, params):
    """
    Construct a heuristic matrix from parameters.
    params is a dict with:
        beta: exponent for inverse distance
        gamma: exponent for savings
        delta: exponent for ring ratio (distance to depot similarity)
    """
    dm = distance_matrix
    n = dm.shape[0]
    
    d = dm.copy()
    np.fill_diagonal(d, 1e9)
    
    beta = max(params.get('beta', 2.0), 0.1)
    gamma = max(params.get('gamma', 1.0), 0.0)
    delta = max(params.get('delta', 1.0), 0.0)
    
    # 1. Inverse Distance
    dist_inv = 1.0 / (d + 1e-9)
    dist_comp = dist_inv ** beta
    
    # 2. Savings
    d_i0 = dm[:, 0]
    d_j0 = dm[0, :]
    savings = d_i0[:, np.newaxis] + d_j0[np.newaxis, :] - d
    savings_pos = np.maximum(savings, 0.0)
    # Normalize savings to [0, 1] to avoid huge values affecting exponents
    max_sav = np.max(savings_pos)
    if max_sav > 1e-9:
        sav_norm = savings_pos / max_sav
    else:
        sav_norm = np.ones_like(savings)
    sav_comp = (1.0 + sav_norm) ** gamma
    
    # 3. Ring Ratio (Cluster Similarity)
    # Ratio of min distance to depot over max distance to depot
    # 1.0 if same distance, < 1.0 if different
    min_d0 = np.minimum(d_i0[:, np.newaxis], d_j0[np.newaxis, :])
    max_d0 = np.maximum(d_i0[:, np.newaxis], d_j0[np.newaxis, :])
    ring_ratio = min_d0 / (max_d0 + 1e-9)
    ring_comp = ring_ratio ** delta
    
    # Combine multiplicatively
    # h = dist_comp * sav_comp * ring_comp
    # Using logs for numerical stability if needed, but values are bounded
    h = dist_comp * sav_comp * ring_comp
    
    # Normalize
    max_h = np.max(h)
    if max_h > 1e-9:
        h = h / max_h
    else:
        h = np.ones_like(h)
    
    # Ensure all entries are positive
    h = np.maximum(h, 1e-9)
    np.fill_diagonal(h, 0.0)
    
    return h

def heuristics(distance_matrix, coordinates, demands, capacity):
    """
    Compute an edge-attractiveness matrix for ACO on CVRP.
    """
    dm = np.asarray(distance_matrix, dtype=float)
    coords = np.asarray(coordinates, dtype=float)
    D = np.asarray(demands, dtype=float)
    cap = float(capacity)
    n = dm.shape[0]
    
    # Basic validation and fallback
    if dm.shape != (n, n) or not np.isfinite(dm).all() or not np.isfinite(D).all():
        d = dm.copy()
        np.fill_diagonal(d, 1e9)
        return 1.0 / (d + 1e-9)
    
    # Check time
    if _remaining() < 10.0:
        d = dm.copy()
        np.fill_diagonal(d, 1e9)
        return 1.0 / (d + 1e-9)
    
    # Initialize with a reasonable default
    # beta: distance weight, gamma: savings weight, delta: ring weight
    best_params = {'beta': 2.0, 'gamma': 1.0, 'delta': 1.0}
    best_score = float('inf')
    
    # Pre-compute static components to speed up heuristic construction during search
    # dist_inv = 1 / (d + 1e-9)
    d = dm.copy()
    np.fill_diagonal(d, 1e9)
    dist_inv = 1.0 / (d + 1e-9)
    
    d_i0 = dm[:, 0]
    d_j0 = dm[0, :]
    savings = d_i0[:, np.newaxis] + d_j0[np.newaxis, :] - d
    savings_pos = np.maximum(savings, 0.0)
    max_sav = np.max(savings_pos)
    if max_sav > 1e-9:
        sav_norm = savings_pos / max_sav
    else:
        sav_norm = np.ones_like(savings)
    
    min_d0 = np.minimum(d_i0[:, np.newaxis], d_j0[np.newaxis, :])
    max_d0 = np.maximum(d_i0[:, np.newaxis], d_j0[np.newaxis, :])
    ring_ratio = min_d0 / (max_d0 + 1e-9)
    
    def fast_construct(beta, gamma, delta):
        dist_comp = dist_inv ** beta
        sav_comp = (1.0 + sav_norm) ** gamma
        ring_comp = ring_ratio ** delta
        h = dist_comp * sav_comp * ring_comp
        max_h = np.max(h)
        if max_h > 1e-9:
            h = h / max_h
        else:
            h = np.ones_like(h)
        h = np.maximum(h, 1e-9)
        np.fill_diagonal(h, 0.0)
        return h
    
    def evaluate(params):
        h = fast_construct(params['beta'], params['gamma'], params['delta'])
        return _run_aco_simulation(dm, D, cap, h, n_ants=10, n_iter=5, seed=42)
    
    # Random Restart Hill Climbing
    np.random.seed(42)
    best_params = {'beta': 2.0, 'gamma': 1.0, 'delta': 1.0}
    best_score = evaluate(best_params)
    
    # Define search space
    # beta: 0.5 to 5.0
    # gamma: 0.0 to 3.0
    # delta: 0.0 to 3.0
    
    step_sizes = [
        [0.5, 0.5, 0.5],
        [0.2, 0.2, 0.2],
        [0.1, 0.1, 0.1],
        [0.05, 0.05, 0.05],
    ]
    
    # Perform a few random restarts
    for restart in range(3):
        if _remaining() < 20.0:
            break
            
        # Start from best or random
        if restart == 0:
            current_params = best_params.copy()
        else:
            current_params = {
                'beta': np.random.uniform(1.0, 4.0),
                'gamma': np.random.uniform(0.5, 2.5),
                'delta': np.random.uniform(0.5, 2.5)
            }
        
        current_score = evaluate(current_params)
        if current_score < best_score:
            best_score = current_score
            best_params = current_params.copy()
        
        # Local search
        for step_idx, step in enumerate(step_sizes):
            if _remaining() < 10.0:
                break
            
            improved = True
            while improved:
                if _remaining() < 10.0:
                    break
                improved = False
                for key in ['beta', 'gamma', 'delta']:
                    for direction in [-step[step_idx], step[step_idx]]:
                        new_params = current_params.copy()
                        new_val = current_params[key] + direction
                        # Bounds
                        if key == 'beta':
                            if new_val < 0.5: new_val = 0.5
                            if new_val > 5.0: new_val = 5.0
                        elif key == 'gamma':
                            if new_val < 0.0: new_val = 0.0
                            if new_val > 3.0: new_val = 3.0
                        else: # delta
                            if new_val < 0.0: new_val = 0.0
                            if new_val > 3.0: new_val = 3.0
                        
                        new_params[key] = new_val
                        try:
                            score = evaluate(new_params)
                            if score < current_score:
                                current_score = score
                                current_params = new_params
                                improved = True
                                if score < best_score:
                                    best_score = score
                                    best_params = new_params.copy()
                        except Exception:
                            pass
    
    # Construct final heuristic with best parameters
    final_h = fast_construct(best_params['beta'], best_params['gamma'], best_params['delta'])
    
    # Final safety
    final_h = np.maximum(final_h, 1e-9)
    np.fill_diagonal(final_h, 0.0)
    
    return final_h
