"""Heuristic for ACO-CVRP: Parameter-Optimized Heuristic with Voronoi-Density Penalty and Fast ACO Feedback."""
import numpy as np
import time
from scipy.spatial import Voronoi, Delaunay

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
                for j in range(1, n):
                    if not visited[j] and demands[j] <= remaining_load + 1e-9:
                        feasible.append(j)
                
                if not feasible:
                    # Return to depot and reload
                    cost += distance_matrix[node, 0]
                    break
                
                # Calculate probabilities: pheromone[i,j] * heuristic[i,j]
                probs = np.array([pheromone[node, j] * heuristic[node, j] for j in feasible])
                total_prob = np.sum(probs)
                
                if total_prob <= 1e-15:
                    # Fallback: uniform random
                    idx = np.random.randint(len(feasible))
                else:
                    probs = probs / total_prob
                    r = np.random.random()
                    cumulative = np.cumsum(probs)
                    idx = np.searchsorted(cumulative, r)
                    if idx >= len(feasible):
                        idx = len(feasible) - 1
                
                next_node = feasible[idx]
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

def _compute_voronoi_penalty(coordinates, n):
    """
    Compute a Voronoi-based penalty matrix.
    Higher density (smaller Voronoi area) -> lower penalty (higher attractiveness).
    """
    try:
        vor = Voronoi(coordinates)
        # Compute area for each point
        # Voronoi regions can be unbounded; we handle this by clipping to bounding box
        # For simplicity and robustness, we use the convex hull area if available, 
        # or just use the distance to nearest neighbor as a proxy for density.
        # Distance to nearest neighbor is a robust proxy for local density.
        
        # Compute nearest neighbor distance for each point
        nn_dists = np.zeros(n)
        for i in range(n):
            dists = np.linalg.norm(coordinates - coordinates[i], axis=1)
            dists[i] = np.inf
            nn_dists[i] = np.min(dists)
        
        # Smaller nn_dist -> higher density -> lower penalty
        # We want to penalize edges that go to low-density areas? 
        # Actually, the insight says "Voronoi cell scale penalty discourages long-range edges in high-density regions".
        # This suggests that in high-density regions, long edges are bad.
        # So, penalty for edge (i,j) should be higher if i is in a high-density region AND d(i,j) is large?
        # Or simply: prefer edges that are locally consistent with density.
        
        # Let's use a simpler heuristic: 
        # Attractiveness should be higher if the edge length is comparable to the local scale.
        # Local scale = nn_dist.
        # If d(i,j) >> nn_dist(i), it's a "long-range" edge from a dense spot -> penalize.
        # If d(i,j) ~ nn_dist(i), it's a "local" edge -> encourage.
        
        # Let's construct a matrix where H_voronoi[i,j] = exp(-d(i,j) / (k * nn_dist(i)))
        # This makes local edges more attractive in dense regions.
        
        k = 2.0  # Scaling factor
        # Avoid division by zero
        nn_dists_safe = np.maximum(nn_dists, 1e-6)
        
        # Vectorized computation
        # d[i,j] is distance matrix
        # We need d[i,j] / nn_dist[i]
        # This is not a simple outer product.
        
        # For now, let's just return a matrix of 1s as a neutral component, 
        # and integrate it into the parameter search as a multiplicative factor 
        # that is 1.0 if we don't use it, or a specific function if we do.
        
        # Actually, let's pre-compute a "locality score" matrix.
        # L[i,j] = exp(-d[i,j] / (2 * nn_dist[i]))
        # This favors short edges relative to local density.
        
        L = np.ones((n, n))
        for i in range(n):
            for j in range(n):
                if i != j:
                    L[i,j] = np.exp(-distance_matrix[i,j] / (k * nn_dists_safe[i]))
        
        return L
        
    except Exception:
        return np.ones((n, n))

def _construct_heuristic(distance_matrix, coordinates, demands, capacity, params, voronoi_penalty):
    """
    Construct a heuristic matrix from parameters.
    """
    dm = distance_matrix
    n = dm.shape[0]
    
    d = dm.copy()
    np.fill_diagonal(d, 1e9)
    
    # Inverse distance component
    w_dist = max(params.get('w_dist', 1.0), 0.0)
    gamma = max(params.get('gamma', 2.0), 0.0)
    dist_inv = 1.0 / (d + 1e-9)
    dist_comp = dist_inv ** gamma if gamma > 0 else dist_inv
    
    # Savings component: d[i,0] + d[0,j] - d[i,j]
    w_sav = max(params.get('w_sav', 0.5), 0.0)
    delta = max(params.get('delta', 1.0), 0.0)
    d_i0 = dm[:, 0]
    d_j0 = dm[0, :]
    savings = d_i0[:, np.newaxis] + d_j0[np.newaxis, :] - d
    savings_pos = np.maximum(savings, 0.0)
    max_sav = np.max(savings_pos)
    if max_sav < 1e-9:
        sav_norm = np.ones_like(savings)
    else:
        sav_norm = savings_pos / max_sav
    sav_comp = sav_norm ** delta if delta > 0 else sav_norm
    
    # Demand-based component
    w_dem = max(params.get('w_dem', 0.0), 0.0)
    theta = max(params.get('theta', 1.0), 0.0)
    dem_norm = demands / (np.max(demands) + 1e-9)
    dem_comp = np.outer(np.ones(n), dem_norm) ** theta if theta > 0 else np.outer(np.ones(n), dem_norm)
    
    # Voronoi/Density component
    w_vor = max(params.get('w_vor', 0.0), 0.0)
    vor_comp = voronoi_penalty ** 1.0  # Simple use
    
    # Combine multiplicatively
    log_h = (w_dist * np.log(dist_comp + 1e-12) +
             w_sav * np.log(sav_comp + 1e-12) +
             w_dem * np.log(dem_comp + 1e-12) +
             w_vor * np.log(vor_comp + 1e-12))
    h = np.exp(log_h)
    
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
    
    # Compute Voronoi penalty once
    voronoi_penalty = _compute_voronoi_penalty(coords, n)
    
    # Initialize with a reasonable default
    default_params = {
        'w_dist': 2.0,
        'gamma': 2.0,
        'w_sav': 1.0,
        'delta': 1.0,
        'w_dem': 0.5,
        'theta': 1.0,
        'w_vor': 0.5
    }
    
    best_params = default_params.copy()
    best_score = float('inf')
    
    # Coarse grid search
    grid = {
        'w_dist': [1.0, 2.0, 3.0],
        'gamma': [1.5, 2.0, 2.5],
        'w_sav': [0.5, 1.0, 1.5],
        'delta': [1.0, 1.5],
        'w_dem': [0.0, 0.5, 1.0],
        'theta': [1.0],
        'w_vor': [0.0, 0.5, 1.0]
    }
    
    # Evaluate a subset of the grid
    evaluated = 0
    for w_dist in grid['w_dist']:
        if _remaining() < 15.0:
            break
        for gamma in grid['gamma']:
            if _remaining() < 15.0:
                break
            for w_sav in grid['w_sav']:
                if _remaining() < 15.0:
                    break
                for delta in grid['delta']:
                    if _remaining() < 15.0:
                        break
                    for w_dem in grid['w_dem']:
                        if _remaining() < 15.0:
                            break
                        for theta in grid['theta']:
                            if _remaining() < 15.0:
                                break
                            for w_vor in grid['w_vor']:
                                if _remaining() < 15.0:
                                    break
                                params = {
                                    'w_dist': w_dist,
                                    'gamma': gamma,
                                    'w_sav': w_sav,
                                    'delta': delta,
                                    'w_dem': w_dem,
                                    'theta': theta,
                                    'w_vor': w_vor
                                }
                                try:
                                    h = _construct_heuristic(dm, coords, D, cap, params, voronoi_penalty)
                                    score = _run_aco_simulation(dm, D, cap, h, n_ants=10, n_iter=5, seed=42)
                                    if score < best_score:
                                        best_score = score
                                        best_params = params.copy()
                                    evaluated += 1
                                except Exception:
                                    pass
    
    # Fine-tuning: local search around best params
    if _remaining() > 30.0:
        step_mult = 0.3
        for _iter in range(5):
            if _remaining() < 20.0:
                break
            improved = False
            for key in ['w_dist', 'gamma', 'w_sav', 'delta', 'w_dem', 'theta', 'w_vor']:
                if _remaining() < 20.0:
                    break
                for direction in [-step_mult, step_mult]:
                    new_params = best_params.copy()
                    new_val = best_params[key] + direction
                    if new_val < 0.01:
                        new_val = 0.01
                    new_params[key] = new_val
                    try:
                        h = _construct_heuristic(dm, coords, D, cap, new_params, voronoi_penalty)
                        score = _run_aco_simulation(dm, D, cap, h, n_ants=10, n_iter=5, seed=42)
                        if score < best_score:
                            best_score = score
                            best_params = new_params
                            improved = True
                            break
                    except Exception:
                        pass
                if improved:
                    break
            if not improved:
                step_mult *= 0.5
                if step_mult < 0.05:
                    break
    
    # Construct final heuristic with best parameters
    final_h = _construct_heuristic(dm, coords, D, cap, best_params, voronoi_penalty)
    
    # Final safety: ensure all values are positive and diagonal is 0
    final_h = np.maximum(final_h, 1e-9)
    np.fill_diagonal(final_h, 0.0)
    
    return final_h
