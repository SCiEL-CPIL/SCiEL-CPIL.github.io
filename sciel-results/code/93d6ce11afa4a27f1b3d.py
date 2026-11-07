"""Heuristic for ACO-CVRP: Parameter-Optimized Heuristic with Delaunay Triangulation."""
import numpy as np
import time
from scipy.spatial import Delaunay

_T0 = time.time()
_BUDGET = 325.0  # Leave 5s safety margin
_DEADLINE = _T0 + _BUDGET

def _remaining():
    return max(0.0, _DEADLINE - time.time())

def _build_delaunay_mask(coords, n):
    """
    Build a boolean mask of edges from Delaunay triangulation.
    Returns an n x n boolean matrix where True indicates edges in the sparse graph.
    """
    mask = np.zeros((n, n), dtype=bool)
    # Include depot (index 0) in triangulation
    try:
        tri = Delaunay(coords)
        for simplex in tri.simplices:
            for i in range(len(simplex)):
                for j in range(len(simplex)):
                    if i != j:
                        mask[simplex[i], simplex[j]] = True
    except Exception:
        # Fallback: k-nearest neighbors
        from scipy.spatial import cKDTree
        tree = cKDTree(coords)
        k = min(8, n - 1)
        distances, indices = tree.query(coords, k=k+1)
        for i in range(n):
            for j in indices[i]:
                if j != i:
                    mask[i, j] = True
    return mask

def _run_aco_simulation(distance_matrix, demands, capacity, heuristic, n_ants=8, n_iter=4, seed=42):
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

def _construct_heuristic(distance_matrix, coordinates, demands, capacity, params, delaunay_mask=None):
    """
    Construct a heuristic matrix from parameters with optional Delaunay mask.
    params is a dict with:
        w_dist: weight for inverse distance
        w_sav: weight for savings
        w_dem: weight for demand-based proximity
        gamma: exponent for distance inverse
        delta: exponent for savings
        theta: exponent for demand ratio
        w_del: weight for Delaunay edge bonus
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
    
    # Demand-based component: prefer customers with demand close to remaining capacity
    # This is a heuristic that encourages packing routes
    w_dem = max(params.get('w_dem', 0.0), 0.0)
    theta = max(params.get('theta', 1.0), 0.0)
    # For each potential route, we want to prefer customers whose demand fills the vehicle
    # Use a simple proxy: customers with higher demand are more "valuable" per stop
    dem_norm = demands / (np.max(demands) + 1e-9)
    # Higher demand -> slightly more attractive (fewer stops needed)
    dem_comp = np.outer(np.ones(n), dem_norm) ** theta if theta > 0 else np.outer(np.ones(n), dem_norm)
    
    # Delaunay edge component: boost edges in the sparse geometric graph
    w_del = max(params.get('w_del', 1.0), 0.0)
    if delaunay_mask is not None and w_del > 0:
        # Edges in Delaunay triangulation get a boost
        del_comp = np.where(delaunay_mask, 1.0 + w_del, 1.0)
    else:
        del_comp = np.ones_like(dm)
    
    # Combine multiplicatively
    log_h = (w_dist * np.log(dist_comp + 1e-12) +
             w_sav * np.log(sav_comp + 1e-12) +
             w_dem * np.log(dem_comp + 1e-12) +
             np.log(del_comp + 1e-12))
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
    Uses model-and-optimize with Delaunay triangulation for geometric structure.
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
    
    # Build Delaunay mask once
    try:
        delaunay_mask = _build_delaunay_mask(coords, n)
    except Exception:
        delaunay_mask = None
    
    # Check time
    if _remaining() < 10.0:
        d = dm.copy()
        np.fill_diagonal(d, 1e9)
        return 1.0 / (d + 1e-9)
    
    # Initialize with a reasonable default
    default_params = {
        'w_dist': 2.0,
        'gamma': 2.0,
        'w_sav': 1.0,
        'delta': 1.0,
        'w_dem': 0.5,
        'theta': 1.0,
        'w_del': 2.0
    }
    
    best_params = default_params.copy()
    best_score = float('inf')
    
    # Coarse grid search
    grid = {
        'w_dist': [1.0, 2.0, 3.0],
        'gamma': [1.0, 2.0],
        'w_sav': [0.5, 1.0, 2.0],
        'delta': [1.0],
        'w_dem': [0.0, 0.5],
        'theta': [1.0],
        'w_del': [1.0, 2.0, 3.0, 5.0]
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
                            for w_del in grid['w_del']:
                                if _remaining() < 15.0:
                                    break
                                params = {
                                    'w_dist': w_dist,
                                    'gamma': gamma,
                                    'w_sav': w_sav,
                                    'delta': delta,
                                    'w_dem': w_dem,
                                    'theta': theta,
                                    'w_del': w_del
                                }
                                try:
                                    h = _construct_heuristic(dm, coords, D, cap, params, delaunay_mask)
                                    score = _run_aco_simulation(dm, D, cap, h, n_ants=8, n_iter=4, seed=42)
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
            for key in ['w_dist', 'gamma', 'w_sav', 'delta', 'w_dem', 'theta', 'w_del']:
                if _remaining() < 20.0:
                    break
                for direction in [-step_mult, step_mult]:
                    new_params = best_params.copy()
                    new_val = best_params[key] + direction
                    if new_val < 0.01:
                        new_val = 0.01
                    new_params[key] = new_val
                    try:
                        h = _construct_heuristic(dm, coords, D, cap, new_params, delaunay_mask)
                        score = _run_aco_simulation(dm, D, cap, h, n_ants=8, n_iter=4, seed=42)
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
    final_h = _construct_heuristic(dm, coords, D, cap, best_params, delaunay_mask)
    
    # Final safety: ensure all values are positive and diagonal is 0
    final_h = np.maximum(final_h, 1e-9)
    np.fill_diagonal(final_h, 0.0)
    
    return final_h
